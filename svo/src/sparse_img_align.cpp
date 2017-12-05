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
#include <svo/sparse_img_align.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/config.h>
#include <svo/point.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>

namespace svo {

SparseImgAlign::SparseImgAlign(
    int max_level, int min_level, int n_iter,
    Method method, bool display, bool verbose) :
        display_(display),
        max_level_(max_level),
        min_level_(min_level)
{
    n_iter_ = n_iter;
    n_iter_init_ = n_iter_;
    method_ = method;
    verbose_ = verbose;
    eps_ = 0.000001;
}

/**
 * 图像对齐主函数
 * @param ref_frame
 * @param cur_frame
 * @return
 */
size_t SparseImgAlign::run(FramePtr ref_frame, FramePtr cur_frame)
{
    reset();

    if(ref_frame->fts_.empty())
    {
        SVO_WARN_STREAM("SparseImgAlign: no features to track!");
        return 0;
    }

    //!
    ref_frame_ = ref_frame;
    cur_frame_ = cur_frame;

    //! 构造对齐的Patch块，每一行为一个patch块
    ref_patch_cache_ = cv::Mat(ref_frame_->fts_.size(), patch_area_, CV_32F);

    //! 雅克比矩阵维数为(6, fts.size*16)
    jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows*patch_area_);


    visible_fts_.resize(ref_patch_cache_.rows, false); // TODO: should it be reset at each level?

    //! 参考帧到当前帧的变换，初始的时候为单位矩阵
    SE3 T_cur_from_ref(cur_frame_->T_f_w_ * ref_frame_->T_f_w_.inverse());

    //! 最小化光度误差，优化两帧之间的相机位姿
    for(level_=max_level_; level_>=min_level_; --level_)
    {
        mu_ = 0.1;
        jacobian_cache_.setZero();
        have_ref_patch_cache_ = false;
        if(verbose_)
            printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
        optimize(T_cur_from_ref);
    }
    //! 更新位姿当前帧的位姿
    cur_frame_->T_f_w_ = T_cur_from_ref * ref_frame_->T_f_w_;

    //! 返回跟踪到的特征点个数
    return n_meas_/patch_area_;
}

Matrix<double, 6, 6> SparseImgAlign::getFisherInformation()
{
    double sigma_i_sq = 5e-4*255*255; // image noise
    Matrix<double,6,6> I = H_/sigma_i_sq;
    return I;
}

/**
 * 提前计算参考帧的patch块的雅克比矩阵
 */
void SparseImgAlign::precomputeReferencePatches()
{
    //! Step1: 对当前的图像和特征点进行尺度变换
    const int border = patch_halfsize_+1;
    //! 读取当前金字塔层的参考帧图像
    const cv::Mat& ref_img = ref_frame_->img_pyr_.at(level_);

    const int stride = ref_img.cols;
    const float scale = 1.0f/(1<<level_);

    //! 获取参考帧位姿
    const Vector3d ref_pos = ref_frame_->pos();
    const double focal_length = ref_frame_->cam_->errorMultiplier2();
    size_t feature_counter = 0;

    //! 遍历参考帧的每一个Features
    std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
    for(auto it=ref_frame_->fts_.begin(), ite=ref_frame_->fts_.end(); it!=ite; ++it, ++feature_counter, ++visiblity_it)
    {
        //! 确保，参考帧的Feature做了尺度变换之后，对应的Patch还在图像之内
        // check if reference with patch size is within image
        //! 这个地方的尺度变换是什么意思。。。(因为图像做了尺度变换，相应的之前得到的特征点的位置也要做尺度变换)
        const float u_ref = (*it)->px[0]*scale;
        const float v_ref = (*it)->px[1]*scale;
        const int u_ref_i = floorf(u_ref);
        const int v_ref_i = floorf(v_ref);
        if((*it)->point == NULL || u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
            continue;
        *visiblity_it = true;

        // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
        const double depth(((*it)->point->pos_ - ref_pos).norm());
        const Vector3d xyz_ref((*it)->f*depth);

        //！ Step2：求取参考图像在点xyz_ref的雅克比矩阵，对应论文(11)式下面的公式
        //! 估计投影的雅克比矩阵
        // evaluate projection jacobian
        Matrix<double,2,6> frame_jac;
        Frame::jacobian_xyz2uv(xyz_ref, frame_jac);

        //! Step3:求取链式法则中的第一项
        //! 对做了尺度变换的特征点进行双线性插值操作，得到四个权重
        // compute bilateral interpolation weights for reference image
        const float subpix_u_ref = u_ref-u_ref_i;
        const float subpix_v_ref = v_ref-v_ref_i;
        const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
        const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;
        size_t pixel_counter = 0;

        //! 获取参考patch块的首地址
        float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;

        //! 处理单个patch，这里的patch1是在参考图像上的
        for(int y=0; y<patch_size_; ++y)
        {
            //! 获取参考图像Patch块左上角的像素点地址
            uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+y-patch_halfsize_)*stride + (u_ref_i-patch_halfsize_);
            for(int x=0; x<patch_size_; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter)
            {
                //! 得到线性差值之后的灰度值，往参考Patch块中装东西，在求取残差的时候会用到
                // precompute interpolated reference patch color
                *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

                //! 求取x,y方向的梯度值(注意是先插值，再求梯度)
                //! 采用逆向组合算法(inverse compositional): 通过采取梯度总是在相同位置这一性质
                // we use the inverse compositional: thereby we can take the gradient always at the same position
                // get gradient of warped image (~gradient at warped position)
                float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                  -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
                float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                  -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

                //! 参考论文式(11)中雅克比矩阵的链式展开((11)式下面的式子）,这里再乘上梯度就是完整的雅克比矩阵
                // cache the jacobian  1*2*2*6=1*6, 注意雅克比矩阵的维数
                jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) =
                    (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
            }
        }
    }
    have_ref_patch_cache_ = true;
}

//! 计算光度误差
double SparseImgAlign::computeResiduals(const SE3& T_cur_from_ref, bool linearize_system, bool compute_weight_scale)
{
    //! Step1：对当前图像进行尺度变换
    // Warp the (cur)rent image such that it aligns with the (ref)erence image
    const cv::Mat& cur_img = cur_frame_->img_pyr_.at(level_);

    if(linearize_system && display_)
        resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

    //! 计算参考帧特征点Patch的雅克比矩阵
    if(have_ref_patch_cache_ == false)
        precomputeReferencePatches();

    // compute the weights on the first iteration
    std::vector<float> errors;
    if(compute_weight_scale)
        errors.reserve(visible_fts_.size());

    const int stride = cur_img.cols;
    const int border = patch_halfsize_+1;
    const float scale = 1.0f/(1<<level_);
    const Vector3d ref_pos(ref_frame_->pos());
    float chi2 = 0.0;
    size_t feature_counter = 0; // is used to compute the index of the cached jacobian
    std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
    for(auto it=ref_frame_->fts_.begin(); it!=ref_frame_->fts_.end(); ++it, ++feature_counter, ++visiblity_it)
    {
        // check if feature is within image
        if(!*visiblity_it)
            continue;

        //! 计算特征点在当前帧上的投影位置
        //! question: 这个地方为什么不直接使用Feature对应的3D点呢
        // compute pixel location in cur img
        const double depth = ((*it)->point->pos_ - ref_pos).norm();

        const Vector3d xyz_ref((*it)->f*depth);
        const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);

        const Vector2f uv_cur_pyr(cur_frame_->cam_->world2cam(xyz_cur).cast<float>() * scale);
        const float u_cur = uv_cur_pyr[0];
        const float v_cur = uv_cur_pyr[1];
        const int u_cur_i = floorf(u_cur);
        const int v_cur_i = floorf(v_cur);

        //! 判断投影点是否在当前图像帧上
        // check if projection is within the image
        if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;

        //! 双线性差值
        // compute bilateral interpolation weights for the current image
        const float subpix_u_cur = u_cur-u_cur_i;
        const float subpix_v_cur = v_cur-v_cur_i;
        const float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        const float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        const float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        const float w_cur_br = subpix_u_cur * subpix_v_cur;

        //! 获取Patch块的地址, 参考patch在计算残差的时候就已经存入了value
        float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
        size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
        for(int y=0; y<patch_size_; ++y)
        {
            //！ 获取Patch块左上角的像素点地址
            uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_cur_i+y-patch_halfsize_)*stride + (u_cur_i-patch_halfsize_);

            for(int x=0; x<patch_size_; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr)
            {
                //! 求取残差
                // compute residual
                const float intensity_cur = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
                const float res = intensity_cur - (*ref_patch_cache_ptr);

                // used to compute scale for robust cost
                if(compute_weight_scale)
                    errors.push_back(fabsf(res));

                // robustification
                float weight = 1.0;
                if(use_weights_)
                {
                    weight = weight_function_->value(res/scale_);
                }

                //! 残差之和
                chi2 += res*res*weight;
                //! 测量值个数
                n_meas_++;

                if(linearize_system)
                {
                    // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                    const Vector6d J(jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter));
                    H_.noalias() += J*J.transpose()*weight;
                    Jres_.noalias() -= J*res*weight;
                    if(display_)
                        resimg_.at<float>((int) v_cur+y-patch_halfsize_, (int) u_cur+x-patch_halfsize_) = res/255.0;
                }
            }
        }
    }

    //! 第一次迭代时的权重
    // compute the weights on the first iteration
    if(compute_weight_scale && iter_ == 0)
        scale_ = scale_estimator_->compute(errors);

    //! 返回平均残差
    return chi2/n_meas_;
}

int SparseImgAlign::solve()
{
    x_ = H_.ldlt().solve(Jres_);
    if((bool) std::isnan((double) x_[0]))
        return 0;
    return 1;
}

void SparseImgAlign::update(
    const ModelType& T_curold_from_ref,
    ModelType& T_curnew_from_ref)
{
    //! Question: 这个地方不应该取负号啊
    T_curnew_from_ref =  T_curold_from_ref * SE3::exp(-x_);
}

void SparseImgAlign::startIteration()
{}

void SparseImgAlign::finishIteration()
{
    if(display_)
    {
        cv::namedWindow("residuals", CV_WINDOW_AUTOSIZE);
        cv::imshow("residuals", resimg_*10);
        cv::waitKey(0);
    }
}

} // namespace svo

