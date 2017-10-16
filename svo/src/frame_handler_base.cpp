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

#include <vikit/abstract_camera.h>
#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <svo/frame_handler_base.h>
#include <svo/config.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/map.h>
#include <svo/point.h>

namespace svo
{

// definition of global and static variables which were declared in the header
#ifdef SVO_TRACE
vk::PerformanceMonitor* g_permon = NULL;
#endif

FrameHandlerBase::FrameHandlerBase() :
  stage_(STAGE_PAUSED),
  set_reset_(false),
  set_start_(false),
  acc_frame_timings_(10),
  acc_num_obs_(10),
  num_obs_last_(0),
  tracking_quality_(TRACKING_INSUFFICIENT)
{
#ifdef SVO_TRACE
  // Initialize Performance Monitor
  g_permon = new vk::PerformanceMonitor();
  g_permon->addTimer("pyramid_creation");
  g_permon->addTimer("sparse_img_align");
  g_permon->addTimer("reproject");
  g_permon->addTimer("reproject_kfs");
  g_permon->addTimer("reproject_candidates");
  g_permon->addTimer("feature_align");
  g_permon->addTimer("pose_optimizer");
  g_permon->addTimer("point_optimizer");
  g_permon->addTimer("local_ba");
  g_permon->addTimer("tot_time");
  g_permon->addLog("timestamp");
  g_permon->addLog("img_align_n_tracked");
  g_permon->addLog("repr_n_mps");
  g_permon->addLog("repr_n_new_references");
  g_permon->addLog("sfba_thresh");
  g_permon->addLog("sfba_error_init");
  g_permon->addLog("sfba_error_final");
  g_permon->addLog("sfba_n_edges_final");
  g_permon->addLog("loba_n_erredges_init");
  g_permon->addLog("loba_n_erredges_fin");
  g_permon->addLog("loba_err_init");
  g_permon->addLog("loba_err_fin");
  g_permon->addLog("n_candidates");
  g_permon->addLog("dropout");
  g_permon->init(Config::traceName(), Config::traceDir());
#endif

  SVO_INFO_STREAM("SVO initialized");
}

FrameHandlerBase::~FrameHandlerBase()
{
  SVO_INFO_STREAM("SVO destructor invoked");
#ifdef SVO_TRACE
  delete g_permon;
#endif
}

//! 当新的图像帧最开始被处理的执行的函数
bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
    //! 针对第一帧函数
    if(set_start_)
    {
        resetAll();
        stage_ = STAGE_FIRST_FRAME;
    }

    if(stage_ == STAGE_PAUSED)
        return false;

    SVO_LOG(timestamp);
    SVO_DEBUG_STREAM("New Frame");
    SVO_START_TIMER("tot_time");
    timer_.start();

    // some cleanup from last iteration, can't do before because of visualization
    map_.emptyTrash();
    return true;
}

//! 当某一帧图像快要处理结束的时候进入该函数
int FrameHandlerBase::finishFrameProcessingCommon(
    const size_t update_id,
    const UpdateResult dropout,
    const size_t num_observations)
{
    SVO_DEBUG_STREAM("Frame: "<<update_id<<"\t fps-avg = "<< 1.0/acc_frame_timings_.getMean()<<"\t nObs = "<<acc_num_obs_.getMean());
    SVO_LOG(dropout);

    //！ 为了计算帧率
    // save processing time to calculate fps
    acc_frame_timings_.push_back(timer_.stop());

    if(stage_ == STAGE_DEFAULT_FRAME)
        acc_num_obs_.push_back(num_observations);

    num_obs_last_ = num_observations;
    SVO_STOP_TIMER("tot_time");

#ifdef SVO_TRACE
    g_permon->writeToFile();
    {
        boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
        size_t n_candidates = map_.point_candidates_.candidates_.size();
        SVO_LOG(n_candidates);
    }
#endif

    if(dropout == RESULT_FAILURE && (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
    {
        stage_ = STAGE_RELOCALIZING;
        tracking_quality_ = TRACKING_INSUFFICIENT;
    }

    //! tracking效果不好
    else if (dropout == RESULT_FAILURE)
        resetAll();
    if(set_reset_)
        resetAll();

    return 0;
}

//! 复位tracking
void FrameHandlerBase::resetCommon()
{
    map_.reset();
    stage_ = STAGE_PAUSED;
    set_reset_ = false;
    set_start_ = false;
    tracking_quality_ = TRACKING_INSUFFICIENT;
    num_obs_last_ = 0;
    SVO_INFO_STREAM("RESET");
}

//! 根据最终优化后观测到的3D点的个数，设定tracking的效果
void FrameHandlerBase::setTrackingQuality(const size_t num_observations)
{
    tracking_quality_ = TRACKING_GOOD;
    if(num_observations < Config::qualityMinFts())
    {
        SVO_WARN_STREAM_THROTTLE(0.5, "Tracking less than "<< Config::qualityMinFts() <<" features!");
        tracking_quality_ = TRACKING_INSUFFICIENT;
    }

    //! 如果相邻两帧观测到的3D点差值过大，则tracking不充分
    const int feature_drop = static_cast<int>(std::min(num_obs_last_, Config::maxFts())) - num_observations;
    if(feature_drop > Config::qualityMaxFtsDrop())
    {
        SVO_WARN_STREAM("Lost "<< feature_drop <<" features!");
        tracking_quality_ = TRACKING_INSUFFICIENT;
    }
}

//! 比较3D点优化的时间戳
bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
    return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

/**
 *
 * @param frame         当前帧
 * @param max_n_pts     最大特征点树  20
 * @param max_iter      最大迭代次数  5
 */
void FrameHandlerBase::optimizeStructure(FramePtr frame, size_t max_n_pts, int max_iter)
{
    //! 统计当前帧观测到的比较好的3D点
    //! 这里定义了一个双端队列，可以头插尾插，头删尾删
    deque<Point*> pts;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point != NULL)
        pts.push_back((*it)->point);
    }

    //! 当特征点数大于20的时候，只处理20个，小于20个的时候处理pts.size()个
    //! 只处理20个是不是有点少
    max_n_pts = min(max_n_pts, pts.size());
    nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), ptLastOptimComparator);
    for(deque<Point*>::iterator it=pts.begin(); it!=pts.begin()+max_n_pts; ++it)
    {
        (*it)->optimize(max_iter);
        //! Ques: 这个地方的时间戳有什么作用
        (*it)->last_structure_optim_ = frame->id_;
    }
}


} // namespace svo
