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
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

//！ 初始化添加第一帧
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
    reset();
    //！检测当前帧的角点
    detectFeatures(frame_ref, px_ref_, f_ref_);

    //！ 如果第一帧检测到的角点数小于100，则失败
    if(px_ref_.size() < 100)
    {
        SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
        return FAILURE;
    }
    frame_ref_ = frame_ref;
    px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
    return SUCCESS;
}

//! 初始化添加第二帧
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
    trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
    SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

    //! 如果跟踪到的角点个数小于阈值，则返回
    if(disparities_.size() < Config::initMinTracked())
        return FAILURE;

    //! 求取跟踪到角点的像素差的中值
    double disparity = vk::getMedian(disparities_);
    SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
    //! 若中值过小，则直接退出即可，即两帧的视差角太小。
    if(disparity < Config::initMinDisparity())
        return NO_KEYFRAME;

    //! 计算单应性矩阵，求取内点
    computeHomography(f_ref_, f_cur_, frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(), inliers_, xyz_in_cur_, T_cur_from_ref_);
    SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

    //! 内点数过少，则返回
    if(inliers_.size() < Config::initMinInliers())
    {
        SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
        return FAILURE;
    }

    // Rescale the map such that the mean scene depth is equal to the specified scale
    //! 添加3D点的深度
    vector<double> depth_vec;
    for(size_t i=0; i<xyz_in_cur_.size(); ++i)
        depth_vec.push_back((xyz_in_cur_[i]).z());

    //! Ques：这个地方的尺度变换
    //! 获取深度值的中值
    double scene_depth_median = vk::getMedian(depth_vec);
    //! 将深度值转换到指定的尺度下
    double scale = Config::mapScale()/scene_depth_median;
    //! 世界坐标系到当前帧的变换 cRf * fRw
    frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
    //! 为位移变换添加尺度
    frame_cur->T_f_w_.translation() = -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos()
                                      + scale*(frame_cur->pos() - frame_ref_->pos()));

    // For each inlier create 3D point and add feature in both frames
    //! 当前帧变换到世界坐标系
    SE3 T_world_cur = frame_cur->T_f_w_.inverse();
    //! 对每个内点创建3D点，设置特征，添加到这两帧当中
    for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
    {
        Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
        Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);

        //! 该内点离图像的边界至少有10个像素点，且3D点的深度值要大于0
        if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
        {
            Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
            Point* new_point = new Point(pos);

            //! 为两帧添加特征点
            Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
            frame_cur->addFeature(ftr_cur);
            new_point->addFrameRef(ftr_cur);

            Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
            frame_ref_->addFeature(ftr_ref);
            new_point->addFrameRef(ftr_ref);
        }
    }
    return SUCCESS;
}

//! 清除当前帧的角点和信息
void KltHomographyInit::reset()
{
    px_cur_.clear();
    frame_ref_.reset();
}

//! 检测当前图像帧上的Fast角点
void detectFeatures(FramePtr frame, vector<cv::Point2f>& px_vec, vector<Vector3d>& f_vec)
{
    //! 初始化角点检测器，并提取角点
    Features new_features;
    feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
    detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

    // now for all maximum corners, initialize a new seed
    px_vec.clear();
    px_vec.reserve(new_features.size());

    f_vec.clear();
    f_vec.reserve(new_features.size());

    //! 存入关键点和关键点的单位方向向量
    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
        f_vec.push_back(ftr->f);
        delete ftr;
    });
}

/*
 * KLT算法跟踪keypoints
 *
 * disparities：关键点在前后两帧上图像上的差值
 */
void trackKlt(FramePtr frame_ref, FramePtr frame_cur, vector<cv::Point2f>& px_ref,
            vector<cv::Point2f>& px_cur, vector<Vector3d>& f_ref, vector<Vector3d>& f_cur,
            vector<double>& disparities)
{
    const double klt_win_size = 30.0;
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    vector<uchar> status;
    vector<float> error;
    vector<float> min_eig_vec;


    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
    cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
    vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
    vector<Vector3d>::iterator f_ref_it = f_ref.begin();

    f_cur.clear();
    f_cur.reserve(px_cur.size());
    disparities.clear();
    disparities.reserve(px_cur.size());

    //! 剔除跟踪不好的点
    for(size_t i=0; px_ref_it != px_ref.end(); ++i)
    {
        if(!status[i])
        {
            px_ref_it = px_ref.erase(px_ref_it);
            px_cur_it = px_cur.erase(px_cur_it);
            f_ref_it = f_ref.erase(f_ref_it);
            continue;
        }
        f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
        disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
        ++px_ref_it;
        ++px_cur_it;
        ++f_ref_it;
    }
}

/**
 * 计算单应性矩阵
 * @param f_ref
 * @param f_cur
 * @param focal_length
 * @param reprojection_threshold
 * @param inliers                   计算出的内点标志
 * @param xyz_in_cur                三角化之后的3D点
 * https://github.com/uzh-rpg/rpg_vikit/blob/10871da6d84c8324212053c40f468c6ab4862ee0/vikit_common/src/math_utils.cpp#L83
 * @param 由H矩阵解算出的位姿
 */
void computeHomography(const vector<Vector3d>& f_ref, const vector<Vector3d>& f_cur,
            double focal_length, double reprojection_threshold,
            vector<int>& inliers, vector<Vector3d>& xyz_in_cur, SE3& T_cur_from_ref)
{
    //! 全部投影到单位平面上
    vector<Vector2d > uv_ref(f_ref.size());
    vector<Vector2d > uv_cur(f_cur.size());
    for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
    {
        uv_ref[i] = vk::project2d(f_ref[i]);
        uv_cur[i] = vk::project2d(f_cur[i]);
    }

    vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
    Homography.computeSE3fromMatches();
    vector<int> outliers;

    //! 三角化恢复出3D点，
    //! 这个地方的内外点是由计算出内外点，是由上面计算出的位姿做重投影误差判断出来的，两个(参考帧和当前帧上)
    //! 为什么不使用Ransac计算H矩阵的时候，计算外点呢。
    vk::computeInliers(f_cur, f_ref, Homography.T_c2_from_c1.rotation_matrix(),
                       Homography.T_c2_from_c1.translation(),
                       reprojection_threshold, focal_length,
                       xyz_in_cur, inliers, outliers);

    //! 通过H矩阵恢复出两帧位姿
    T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
