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
#include <svo/pose_optimizer.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>

namespace svo {
namespace pose_optimizer {

/**
 *  基于高斯牛顿的重投影误差优化
 * @param reproj_thresh     重投影误差阈值
 * @param n_iter            优化最大迭代次数
 * @param verbose
 * @param frame             当前图像帧
 * @param estimated_scale
 * @param error_init
 * @param error_final       最终的重投影误差
 * @param num_obs           最终的观测量，BA的边
 */
void optimizeGaussNewton(const double reproj_thresh, const size_t n_iter,
                         const bool verbose, FramePtr& frame,
                         double& estimated_scale, double& error_init,
                         double& error_final, size_t& num_obs)
{
    //! Step1: 初始化
    // init
    double chi2(0.0);
    vector<double> chi2_vec_init, chi2_vec_final;
    vk::robust_cost::TukeyWeightFunction weight_function;
    SE3 T_old(frame->T_f_w_);
    Matrix6d A;
    Vector6d b;

    // compute the scale of the error for robust estimation
    //! Step2：计算重投影误差
    std::vector<float> errors;
    errors.reserve(frame->fts_.size());
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL)
            continue;

        //! 将3D点变换到当前帧坐标系下
        Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);
        //! 这种尺度变换的原理是什么呢
        e *= 1.0 / (1<<(*it)->level);

        errors.push_back(e.norm());
    }
    if(errors.empty())
        return;

    //! 通过计算平均误差，估算尺度
    vk::robust_cost::MADScaleEstimator scale_estimator;
    estimated_scale = scale_estimator.compute(errors);

    num_obs = errors.size();
    chi2_vec_init.reserve(num_obs);
    chi2_vec_final.reserve(num_obs);
    double scale = estimated_scale;

    //! Step3: 迭代优化当前帧位姿
    for(size_t iter=0; iter<n_iter; iter++)
    {
        //! Ques：为什么迭代次数是5的时候要重新计算尺度
        // overwrite scale
        if(iter == 5)
            scale = 0.85/frame->cam_->errorMultiplier2();

        b.setZero();
        A.setZero();
        double new_chi2(0.0);

        //! Step3.1：计算残差
        // compute residual
        for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
        {
            if((*it)->point == NULL)
                continue;
            Matrix26d J;

            //！ 当前帧坐标系下的3D点
            Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);

            //! 计算 de/dδT
            Frame::jacobian_xyz2uv(xyz_f, J);

            //! 相机归一化平面上的误差
            Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
            double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
            e *= sqrt_inv_cov;

            if(iter == 0)
                chi2_vec_init.push_back(e.squaredNorm()); // just for debug
            J *= sqrt_inv_cov;

            //！Ques：这个权重计算的原理是什么
            double weight = weight_function.value(e.norm()/scale);

            //! 计算高斯牛顿的增量方程，AdT = b
            A.noalias() += J.transpose()*J*weight;
            b.noalias() -= J.transpose()*e*weight;
            new_chi2 += e.squaredNorm()*weight;
        }

        //! 求解更新的位姿
        // solve linear system
        //! 求解的表达是李代数的形式
        const Vector6d dT(A.ldlt().solve(b));

        // check if error increased
        if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
        {
            if(verbose)
                std::cout << "it " << iter
            << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
            frame->T_f_w_ = T_old; // roll-back
            break;
        }

        // update the model
        SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
        T_old = frame->T_f_w_;
        frame->T_f_w_ = T_new;
        chi2 = new_chi2;
        if(verbose)
            std::cout << "it " << iter
        << "\t Success \t new_chi2 = " << new_chi2
        << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;

        // stop when converged
        if(vk::norm_max(dT) <= EPS)
            break;
    }

    //！Step4：求取协方差矩阵(信息矩阵的逆)
    // Set covariance as inverse information matrix. Optimistic estimator!
    const double pixel_variance=1.0;
    frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(), 2)).inverse();

    //！Step5：移除以优化后位姿投影，误差还比较大的Point
    // Remove Measurements with too large reprojection error
    double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
    size_t n_deleted_refs = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL)
            continue;
        Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);
        double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
        e *= sqrt_inv_cov;
        chi2_vec_final.push_back(e.squaredNorm());
        if(e.norm() > reproj_thresh_scaled)
        {
            // we don't need to delete a reference in the point since it was not created yet
            (*it)->point = NULL;
            ++n_deleted_refs;
        }
    }

    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())
        error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
    if(!chi2_vec_final.empty())
        error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

    estimated_scale *= frame->cam_->errorMultiplier2();
    if(verbose)
        std::cout << "n deleted obs = " << n_deleted_refs
    << "\t scale = " << estimated_scale
    << "\t error init = " << error_init
    << "\t error end = " << error_final << std::endl;
    num_obs -= n_deleted_refs;
}

} // namespace pose_optimizer
} // namespace svo
