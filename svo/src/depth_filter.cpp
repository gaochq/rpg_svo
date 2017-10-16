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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo {

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

//! 种子点初始化
Seed::Seed(Feature* ftr, float depth_mean, float depth_min) :
    batch_id(batch_counter),            //!创建该seed的关键帧id
    id(seed_counter++),                 //!种子id
    ftr(ftr),                           //!要恢复深度的特征点
    a(10),                              //!Beta分布的参数
    b(10),                              //!Beta分布的参数
    mu(1.0/depth_mean),                 //!正太分布的初始均值，设置为平均深度的倒数，逆深度表示
    z_range(1.0/depth_min),             //!最大逆深度
    sigma2(z_range*z_range/36)          //!正态分布的协方差
{}

//! 深度滤波器初始化
DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
    feature_detector_(feature_detector),            //!
    seed_converged_cb_(seed_converged_cb),          //！
    seeds_updating_halt_(false),                    //！
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0)
{}

DepthFilter::~DepthFilter()
{
    stopThread();
    SVO_INFO_STREAM("DepthFilter destructed.");
}

//! 启动深度滤波线程
void DepthFilter::startThread()
{
    thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
    SVO_INFO_STREAM("DepthFilter stop thread invoked.");
    if(thread_ != NULL)
    {
        SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
        seeds_updating_halt_ = true;
        thread_->interrupt();
        thread_->join();
        thread_ = NULL;
    }
}

void DepthFilter::addFrame(FramePtr frame)
{
    if(thread_ != NULL)
    {
        {
            lock_t lock(frame_queue_mut_);
            if(frame_queue_.size() > 2)
            frame_queue_.pop();
            frame_queue_.push(frame);
        }
        seeds_updating_halt_ = false;
        frame_queue_cond_.notify_one();
    }
    else
        updateSeeds(frame);
}

//! 初始化新一帧的深度滤波器
//！ QUES：新的一帧的3D点的是怎么来的
void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
{
    new_keyframe_min_depth_ = depth_min;
    new_keyframe_mean_depth_ = depth_mean;

    //! 深度滤波器的线程正在进行
    if(thread_ != NULL)
    {
        new_keyframe_ = frame;
        new_keyframe_set_ = true;
        seeds_updating_halt_ = true;
        frame_queue_cond_.notify_one();
    }
    //! 深度滤波器还没有开始
    else
        initializeSeeds(frame);
}

//! 初始化深度滤波器的种子点
void DepthFilter::initializeSeeds(FramePtr frame)
{
    //！做一次特征检测
    Features new_features;
    feature_detector_->setExistingFeatures(frame->fts_);
    feature_detector_->detect(frame.get(), frame->img_pyr_,
                              Config::triangMinCornerScore(), new_features);

    // initialize a seed for every new feature
    //! 把检测到的特征点都当做种子点
    seeds_updating_halt_ = true;
    lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
    ++Seed::batch_counter;
    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
    });

    if(options_.verbose)
        SVO_INFO_STREAM("DepthFilter: Initialized "<<new_features.size()<<" new seeds");
    seeds_updating_halt_ = false;
}

//! 在深度滤波器中删除关键帧frame
void DepthFilter::removeKeyframe(FramePtr frame)
{
    //! 主要是删除该帧对应的种子点
    seeds_updating_halt_ = true;
    lock_t lock(seeds_mut_);
    std::list<Seed>::iterator it=seeds_.begin();
    size_t n_removed = 0;
    while(it!=seeds_.end())
    {
        if(it->ftr->frame == frame.get())
        {
            it = seeds_.erase(it);
            ++n_removed;
        }
        else
            ++it;
    }
    seeds_updating_halt_ = false;
}

void DepthFilter::reset()
{
    seeds_updating_halt_ = true;
    {
        lock_t lock(seeds_mut_);
        seeds_.clear();
    }
    lock_t lock();
    while(!frame_queue_.empty())
        frame_queue_.pop();
    seeds_updating_halt_ = false;

    if(options_.verbose)
        SVO_INFO_STREAM("DepthFilter: RESET.");
}

//! 更新深度滤波器种子的主线程
void DepthFilter::updateSeedsLoop()
{
    while(!boost::this_thread::interruption_requested())
    {
        FramePtr frame;
        {
            //! 等待有新帧的加入
            lock_t lock(frame_queue_mut_);
            while(frame_queue_.empty() && new_keyframe_set_ == false)
                frame_queue_cond_.wait(lock);
            //! 有新帧到来
            if(new_keyframe_set_)
            {
                new_keyframe_set_ = false;
                seeds_updating_halt_ = false;
                clearFrameQueue();
                frame = new_keyframe_;
            }
            else
            {
                frame = frame_queue_.front();
                frame_queue_.pop();
            }
        }
        //！这个地方要注意，在没有新帧加入的时候当前帧是一直被更新种子点，计算深度的。
        //！种子更新
        updateSeeds(frame);

        //! 如果当前帧是关键帧，则将其特征点当做种子点加入
        if(frame->isKeyframe())
            initializeSeeds(frame);
    }
}

//！ 更新所有种子，包括逆深度和不确定度
void DepthFilter::updateSeeds(FramePtr frame)
{
    // update only a limited number of seeds, because we don't have time to do it
    // for all the seeds in every frame!
    size_t n_updates=0, n_failed_matches=0, n_seeds = seeds_.size();
    lock_t lock(seeds_mut_);
    std::list<Seed>::iterator it=seeds_.begin();

    //! Step1:获取单位像素引起的角度和深度的不确定性
    //! 参考文献[3]式（16）  δβ = 2*atan(1/2f)
    const double focal_length = frame->cam_->errorMultiplier2();
    double px_noise = 1.0;
    double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

    //! Step2:遍历所有种子点
    while( it!=seeds_.end())
    {
        //! 当有新帧到来的时候，不处理种子点
        // set this value true when seeds updating should be interrupted
        if(seeds_updating_halt_)
            return;

        //! Step2.1: 判断种子点在多次估计之后还没有收敛，则直接将其剔除
        // check if seed is not already too old
        if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs)
        {
            it = seeds_.erase(it);
            continue;
        }

        //！Step2.2：判断种子点在当前帧图像时是否可被观测
        // check if point is visible in the current image
        //！求取种子的创建帧和当前帧的位姿关系
        SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
        //! 转到当前帧的相机坐标系下
        const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f) );
        //! 点在相机后面
        if(xyz_f.z() < 0.0)
        {
            ++it; // behind the camera
            continue;
        }
        //!点不在相机投影平面上
        if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>()))
        {
            ++it; // point does not project in image
            continue;
        }

        //! Step3: 极线搜索寻找匹配点位置
        // we are using inverse depth coordinates
        //! 确定投影极线的最大和最小逆深度值
        float z_inv_min = it->mu + sqrt(it->sigma2);
        float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
        double z;
        //! 在当前帧中使用极线搜索当前种子点的匹配点,若极线搜索没有匹配点，则直接返回
        if(!matcher_.findEpipolarMatchDirect(*it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
        {
            it->b++; // increase outlier probability when no match was found
            ++it;
            ++n_failed_matches;
            continue;
        }

        //! Step4：计算不确定度
        // compute tau
        double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
        double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

        //! Step5：更新种子的后验参数
        // update the estimate
        updateSeed(1./z, tau_inverse*tau_inverse, &*it);
        ++n_updates;

        //！ 如果该帧是关键帧，则将将该特征点对应的网格标记为占有，在该网格中不再生成新的点
        //!  在种子更新函数DepthFilter::updateSeedsLoop()中有说明
        if(frame->isKeyframe())
        {
            // The feature detector should not initialize new seeds close to this location
            feature_detector_->setGridOccpuancy(matcher_.px_cur_);
        }

        //! Step6:如果种子点已经收敛，则移除该种子点，并初始化一个新的种子点
        // if the seed has converged, we initialize a new candidate point and remove the seed
        if(sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
        {
            assert(it->ftr->point == NULL); // TODO this should not happen anymore
            //! 将该种子点转到世界坐标系下
            Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0/it->mu)));
            Point* point = new Point(xyz_world, it->ftr);
            it->ftr->point = point;
            /* FIXME it is not threadsafe to add a feature to the frame here.
            if(frame->isKeyframe())
            {
            Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
            ftr->point = point;
            point->addFrameRef(ftr);
            frame->addFeature(ftr);
            it->ftr->frame->addFeature(it->ftr);
            }
            else
            */
            {
                seed_converged_cb_(point, it->sigma2); // put in candidate list
            }
            it = seeds_.erase(it);
        }
        else if(isnan(z_inv_min))
        {
            SVO_WARN_STREAM("z_min is NaN");
            it = seeds_.erase(it);
        }
        else
            ++it;
    }
}

//! 清除深度滤波器的帧队列
void DepthFilter::clearFrameQueue()
{
    while(!frame_queue_.empty())
        frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
    lock_t lock(seeds_mut_);
    for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
    {
        if (it->ftr->frame == frame.get())
            seeds.push_back(*it);
    }
}

/**
 *  更新种子点
 * @param x         种子点在当前帧下的逆深度
 * @param tau2      测量不确定度
 * @param seed      种子点
 *
 * 对应参考文献[2]，Supplementary material式19-式26
 */
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
    //!
    float norm_scale = sqrt(seed->sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    //! 建立一个正太分布
    boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
    float s2 = 1./(1./seed->sigma2 + 1./tau2);
    float m = s2*(seed->mu/seed->sigma2 + x/tau2);
    float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
    float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
    float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
              + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*seed->mu;
    seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
    seed->mu = mu_new;
    seed->a = (e-f)/(f-e/f);
    seed->b = seed->a*(1.0f-f)/f;
}

/**
 * 计算测量的不确定度，对照文献[3]中III.C部分
 * @param T_ref_cur         参考帧到当前帧的变换关系
 * @param f                 种子点的单位方向向量
 * @param z                 三角化的深度值
 * @param px_error_angle    单位像素引起的深度误差
 * @return                  深度测量的不确定性
 *
 *
 */
double DepthFilter::computeTau(const SE3& T_ref_cur, const Vector3d& f,
                               const double z, const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

} // namespace svo
