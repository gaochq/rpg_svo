// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
    initialize();
}

//! 单目里程计的初始化
void FrameHandlerMono::initialize()
{
    //! 构造角点检测器和深度滤波器
    feature_detection::DetectorPtr feature_detector(
        new feature_detection::FastDetector(cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
    DepthFilter::callback_t depth_filter_cb = boost::bind(
        &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
    //! 深度滤波器初始化
    depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
    depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

//! 处理新的图像
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
    if(!startFrameProcessingCommon(timestamp))
        return;

    // some cleanup from last iteration, can't do before because of visualization
    core_kfs_.clear();
    overlap_kfs_.clear();

    //! 建立新的图像帧
    // create new frame
    SVO_START_TIMER("pyramid_creation");
    new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
    SVO_STOP_TIMER("pyramid_creation");

    // process frame
    UpdateResult res = RESULT_FAILURE;
    if(stage_ == STAGE_DEFAULT_FRAME)
        res = processFrame();
    else if(stage_ == STAGE_SECOND_FRAME)
        res = processSecondFrame();
    else if(stage_ == STAGE_FIRST_FRAME)
        res = processFirstFrame();
    else if(stage_ == STAGE_RELOCALIZING)
        res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

    // set last frame
    last_frame_ = new_frame_;
    new_frame_.reset();

    //! 结束处理
    // finish processing
    finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

//! 处理第一帧图像
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
    //! 将第一帧作为原点
    new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());

    //! 处理第一帧图像，并判断其角点个数是否大于100，并以此判断其是否为关键帧，若不是，则直接返回。
    if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
        return RESULT_NO_KEYFRAME;
    new_frame_->setKeyframe();
    map_.addKeyframe(new_frame_);
    stage_ = STAGE_SECOND_FRAME;
    SVO_INFO_STREAM("Init: Selected first frame.");
    return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
    initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);

    if(res == initialization::FAILURE)
        return RESULT_FAILURE;
    else if(res == initialization::NO_KEYFRAME)
        return RESULT_NO_KEYFRAME;

    // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
    ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

    //! 将第二帧设为关键帧，已经满足了角点个数大于阈值，视差角大于阈值的判断。
    new_frame_->setKeyframe();

    //! 求取该帧特征点在相机坐标系下深度值的平均值和最小值
    double depth_mean, depth_min;
    frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

    //! 将关键帧添加到深度估计的算法队列中
    depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

    //! 将关键帧添加到地图中。
    // add frame to map
    map_.addKeyframe(new_frame_);
    stage_ = STAGE_DEFAULT_FRAME;
    klt_homography_init_.reset();
    SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
    return RESULT_IS_KEYFRAME;
}

//! 正常的跟踪处理函数
FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
    // Set initial pose TODO use prior
    //! 本帧的初始位姿使用上一帧的位姿
    new_frame_->T_f_w_ = last_frame_->T_f_w_;

    //!Step1：对应论文IV.A 光度误差的Image Alignmen
    //! 最小化光度误差，优化相机位姿，返回成功跟踪的特征点个数
    // sparse image align
    SVO_START_TIMER("sparse_img_align");
    SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
    size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
    SVO_STOP_TIMER("sparse_img_align");
    SVO_LOG(img_align_n_tracked);
    SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);


    //！Step2：对应论文IV.B  光度误差的Feature Alignment
    //！ 将本帧与其他帧共视的mappoint投影到本帧上，且秉承这每个格子只有一个最佳投影点的原则
    //！为下一步利用重投影误差，进一步优化相机位姿做准备。
    // map reprojection & feature alignment
    SVO_START_TIMER("reproject");
    reprojector_.reprojectMap(new_frame_, overlap_kfs_);
    SVO_STOP_TIMER("reproject");
    //! 成功投影的个数
    const size_t repr_n_new_references = reprojector_.n_matches_;
    //! 当前帧中和其他帧共视的mappoint总数
    const size_t repr_n_mps = reprojector_.n_trials_;
    SVO_LOG2(repr_n_mps, repr_n_new_references);
    SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
    //! 若成功投影的个数小于阈值，则直接复位，防止大的位姿变动
    if(repr_n_new_references < Config::qualityMinFts())
    {
        SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
        new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        tracking_quality_ = TRACKING_INSUFFICIENT;
        return RESULT_FAILURE;
    }

    //! Step3：对应论文IV.C  基于重投影误差的Bundle Adjustment，优化相机位姿
    //! Step3.1: 优化相机位姿
    // pose optimization
    SVO_START_TIMER("pose_optimizer");
    size_t sfba_n_edges_final;
    double sfba_thresh, sfba_error_init, sfba_error_final;
    pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
    SVO_STOP_TIMER("pose_optimizer");
    SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
    SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
    SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
    //! 如果最终的观测少于20，tracking失败
    if(sfba_n_edges_final < 20)
        return RESULT_FAILURE;

    //! Step3.2：优化3D点坐标
    // structure optimization
    SVO_START_TIMER("point_optimizer");
    optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
    SVO_STOP_TIMER("point_optimizer");

    // select keyframe
    //！Step4：设定tracking效果
    core_kfs_.insert(new_frame_);
    setTrackingQuality(sfba_n_edges_final);
    //! 如果跟踪效果不好，则将当前帧的位姿置为上一帧位姿
    if(tracking_quality_ == TRACKING_INSUFFICIENT)
    {
        new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        return RESULT_FAILURE;
    }

    //！Step5：为下一次的投影做好准备
    //! Step5.1：计算本帧3D点在相机坐标系下的最小深度和平均深度
    double depth_mean, depth_min;
    frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
    if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
    {
        depth_filter_->addFrame(new_frame_);
        return RESULT_NO_KEYFRAME;
    }

    //！ 将本帧设定为关键帧，并设定5个关键点
    new_frame_->setKeyframe();
    SVO_DEBUG_STREAM("New keyframe selected.");

    // new keyframe selected
    //! Step5.2：选择新的关键帧
    for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
        if((*it)->point != NULL)
            (*it)->point->addFrameRef(*it);
    //! Step5.3：在map中加入候选点，方便下一帧的投影
    map_.point_candidates_.addCandidatePointToFrame(new_frame_);

    // optional bundle adjustment
    #ifdef USE_BUNDLE_ADJUSTMENT
    if(Config::lobaNumIter() > 0)
    {
        SVO_START_TIMER("local_ba");
        setCoreKfs(Config::coreNKfs());
        size_t loba_n_erredges_init, loba_n_erredges_fin;
        double loba_err_init, loba_err_fin;
        ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                    loba_n_erredges_init, loba_n_erredges_fin,
                    loba_err_init, loba_err_fin);
        SVO_STOP_TIMER("local_ba");
        SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
        SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                         "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
    }
    #endif

    //！Step6：初始化新一帧的深度滤波器
    // init new depth-filters
    depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

    // if limited number of keyframes, remove the one furthest apart
    //! 如果地图中的关键帧的数目大于maxNKfs() (10)
    if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
    {
        //! 获取最远的关键帧
        FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());

        //! 在深度滤波器和地图中删除最远的关键帧
        depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
        //! Ques:这个地方的删除有问题
        map_.safeDeleteFrame(furthest_frame);
    }

    // add keyframe to map
    //! Step7: 将关键帧加入到地图当中
    map_.addKeyframe(new_frame_);

    return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
    SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
    if(ref_keyframe == nullptr)
    {
        SVO_INFO_STREAM("No reference keyframe.");
        return RESULT_FAILURE;
    }
    SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                       30, SparseImgAlign::GaussNewton, false, false);
    size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
    if(img_align_n_tracked > 30)
    {
        SE3 T_f_w_last = last_frame_->T_f_w_;
        last_frame_ = ref_keyframe;
        FrameHandlerMono::UpdateResult res = processFrame();
        if(res != RESULT_FAILURE)
        {
            stage_ = STAGE_DEFAULT_FRAME;
            SVO_INFO_STREAM("Relocalization successful.");
        }
        else
            new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
        return res;
    }
    return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
    FramePtr ref_keyframe;
    if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
        return false;
    new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
    UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
    if(res != RESULT_FAILURE)
    {
        last_frame_ = new_frame_;
        return true;
    }
    return false;
}

//! 复位VO
void FrameHandlerMono::resetAll()
{
    resetCommon();
    last_frame_.reset();
    new_frame_.reset();
    core_kfs_.clear();
    overlap_kfs_.clear();
    depth_filter_->reset();
}

//! 将第一帧作为参考帧
void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
    resetAll();
    last_frame_ = first_frame;
    last_frame_->setKeyframe();
    map_.addKeyframe(last_frame_);
    stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
    for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
    {
        Vector3d relpos = new_frame_->w2f(it->first->pos());
        if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
        fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
        fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
            return false;
    }
    return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
    size_t n = min(n_closest, overlap_kfs_.size()-1);
    std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
    std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
