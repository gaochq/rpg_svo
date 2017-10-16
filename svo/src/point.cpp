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

#include <stdexcept>
#include <vikit/math_utils.h>
#include <svo/point.h>
#include <svo/frame.h>
#include <svo/feature.h>
 
namespace svo {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(0),
  v_pt_(NULL),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{}

Point::Point(const Vector3d& pos, Feature* ftr) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(1),
  v_pt_(NULL),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{
  obs_.push_front(ftr);
}

Point::~Point()
{}

//! 添加Point的投影点
void Point::addFrameRef(Feature* ftr)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

Feature* Point::findFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    if((*it)->frame == frame)
      return *it;
  return NULL;    // no keyframe found
}

//! 在3D点的观测中删除在关键帧frame中的feature
bool Point::deleteFrameRef(Frame* frame)
{
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        if((*it)->frame == frame)
        {
            obs_.erase(it);
            return true;
        }
    }
    return false;
}

void Point::initNormal()
{
    assert(!obs_.empty());
    const Feature* ftr = obs_.back();
    assert(ftr->frame != NULL);
    normal_ = ftr->frame->T_f_w_.rotation_matrix().transpose()*(-ftr->f);
    normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
    normal_set_ = true;
}

//! 寻找观测到该特征点的，最小视差角的两帧
bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr) const
{
    // TODO: get frame with same point of view AND same pyramid level!
    //! 得到观察的方向向量
    Vector3d obs_dir(framepos - pos_);
    obs_dir.normalize();

    auto min_it=obs_.begin();
    double min_cos_angle = 0;
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        //! 得到另一个观察的方向向量
        Vector3d dir((*it)->frame->pos() - pos_);
        dir.normalize();

        //！寻找视差角最大的两次观测，余弦值越大，角度越小
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;

    //！ 视差角至少要大于60°
    if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
        return false;
    return true;
}

//! 优化3D点坐标
void Point::optimize(const size_t n_iter)
{
    Vector3d old_point = pos_;
    double chi2 = 0.0;
    Matrix3d A;
    Vector3d b;

    for(size_t i=0; i<n_iter; i++)
    {
        A.setZero();
        b.setZero();
        double new_chi2 = 0.0;

        // compute residuals
        //！ 计算残差
        for(auto it=obs_.begin(); it!=obs_.end(); ++it)
        {
            Matrix23d J;
            //！ 将3D点转到相机坐标系下
            const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
            //! 求取雅克比矩阵和归一化平面上的重投影误差
            Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J);
            const Vector2d e(vk::project2d((*it)->f) - vk::project2d(p_in_f));
            new_chi2 += e.squaredNorm();
            A.noalias() += J.transpose() * J;
            b.noalias() -= J.transpose() * e;
        }

        // solve linear system
        const Vector3d dp(A.ldlt().solve(b));

        // check if error increased
        //! 检查重投影误差是否变大
        if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
        {
            #ifdef POINT_OPTIMIZER_DEBUG
            cout << "it " << i
            << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
            #endif
            pos_ = old_point; // roll-back
            break;
        }

        // update the model
        Vector3d new_point = pos_ + dp;
        old_point = pos_;
        pos_ = new_point;
        chi2 = new_chi2;
        #ifdef POINT_OPTIMIZER_DEBUG
        cout << "it " << i
        << "\t Success \t new_chi2 = " << new_chi2
        << "\t norm(b) = " << vk::norm_max(b)
        << endl;
        #endif

        // stop when converged
        if(vk::norm_max(dp) <= EPS)
            break;
    }
    #ifdef POINT_OPTIMIZER_DEBUG
    cout << endl;
    #endif
}

} // namespace svo
