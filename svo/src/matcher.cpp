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

#include <cstdlib>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/patch_score.h>
#include <svo/matcher.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/feature_alignment.h>

namespace svo {

namespace warp {

/**
 * 计算仿射变换矩阵
 * @param cam_ref       参考帧相机位姿信息
 * @param cam_cur       当前帧相机位姿信息
 * @param px_ref        Feature在最底层金字塔的坐标
 * @param f_ref         Feature的单位方向向量
 * @param depth_ref     3D点距相机的距离(深度值)
 * @param T_cur_ref     ref ===> cur
 * @param level_ref     特征点所在的金字塔层数
 * @param A_cur_ref     计算出的仿射矩阵
 */
void getWarpMatrixAffine(const vk::AbstractCamera& cam_ref, const vk::AbstractCamera& cam_cur,
                        const Vector2d& px_ref, const Vector3d& f_ref,
                        const double depth_ref, const SE3& T_cur_ref,  //! ref ===> cur
                        const int level_ref, Matrix2d& A_cur_ref)
{
    // Compute affine warp matrix A_ref_cur
    const int halfpatch_size = 5;
    //! 3D点在ref下的坐标
    const Vector3d xyz_ref(f_ref*depth_ref);

    //! 构成一个直角三角形，并将其变换到世界坐标系之下。(实际是相机坐标系下)
    Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
    Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));

    //! 将变换后的三角形的另外两个定点，加上深度信息。因为这几个像素点离得近，所以可以认为深度值是一致的
    xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];

    //! 转到当前的图像平面上
    const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));
    const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
    const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));

    //! 如果没有发生仿射变换，这个地方求出的A矩阵，就是一个单位矩阵
    //! 如果只有尺度变换的话，主对角线有值，副对角线为0，副对角线代表了旋转。
    //! 因为没有平移变换，所以最终的仿射矩阵是一个2*2的矩阵
    A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

//! 根据A_cur_ref求取最好的图像尺度，来搜索该patch
int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level)
{
    // Compute patch level in other image
    int search_level = 0;
    double D = A_cur_ref.determinant();
    //！这里行列式的值，和金字塔的level是什么关系？
    while(D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

/**
 * 对patch进行仿射变换
 * @param A_cur_ref     仿射矩阵
 * @param img_ref
 * @param px_ref        特征点
 * @param level_ref     特征点被提取的金字塔层数
 * @param search_level
 * @param halfpatch_size
 * @param patch         变换之后的patch块
 */
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
    const int patch_size = halfpatch_size*2 ;
    const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();

    //! Question: 这里应该本来就没有平移量吧？
    if(isnan(A_ref_cur(0,0)))
    {
        printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
        return;
    }

    // Perform the warp on a larger patch.
    uint8_t* patch_ptr = patch;
    const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
    for (int y=0; y<patch_size; ++y)
    {
        for (int x=0; x<patch_size; ++x, ++patch_ptr)
        {
            //！先做尺度变换
            Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
            px_patch *= (1<<search_level);

            //！ 再做仿射变换之后，加上特征点的坐标
            const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
            //! 判断仿射变换之后的点是否还在图像上，在的话做一次双插
            if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
                *patch_ptr = 0;
            else
                *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        }
    }
}

} // namespace warp

/**
 * 三角化测量深度
 * @param T_search_ref  参考帧到当前帧的SE3
 * @param f_ref         参考帧坐标系下的3D点
 * @param f_cur         当前帧坐标系下的3D点
 * @param depth         估计出的深度值
 * @return
 *
 *  R_cr*f_r*dr +(-) f_c*dc = t_cr
 *  等式里面的加号应该为减号，所以最后的dc为负值，直接取绝对值即可
 *  然后进一步就可以写为下式了
 *
 * |x0 x1| |d0| |t0|
 * |y0 y1| |d1|=|t1|
 * |1   1|      |t2|
 */
bool depthFromTriangulation(const SE3& T_search_ref, const Vector3d& f_ref,
                            const Vector3d& f_cur, double& depth)
{
    Matrix<double,3,2> A;
    A << T_search_ref.rotation_matrix() * f_ref, f_cur;
    const Matrix2d AtA = A.transpose()*A;
    if(AtA.determinant() < 0.000001)
        return false;
    const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
    depth = fabs(depth2[0]);
    return true;
}

//! 获取去掉边界的扩展patch
void Matcher::createPatchFromPatchWithBorder()
{
    //! 获取没有边界的patch的地址
    uint8_t* ref_patch_ptr = patch_;
    for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
    {
        //! 获取行首地址+1
        uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
        for(int x=0; x<patch_size_; ++x)
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
}

//! 通过逆向构造的方法对patch块对齐，进而得到准确的Feature位置(px)
bool Matcher::findMatchDirect(const Point& pt, const Frame& cur_frame, Vector2d& px_cur)
{
    //! Step1:寻找对于3D点pt来说，与当前帧视差角最小的一个共视帧上的投影点(ref_ftr_)
    //! Question：为什么不在一开始就去找这个Feature呢
    if(!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_))
        return false;

    //! 判断特征点经过尺度变换之后是否还在图像上。
    if(!ref_ftr_->frame->cam_->isInFrame(ref_ftr_->px.cast<int>()/(1<<ref_ftr_->level), halfpatch_size_+2, ref_ftr_->level))
        return false;

    //！ Step2：求取仿射变换矩阵A_cur_ref_
    warp::getWarpMatrixAffine(
        *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
        (ref_ftr_->frame->pos() - pt.pos_).norm(),
        cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level, A_cur_ref_
    );

    //! Step3：依据仿射矩阵行列式的值，获取最佳的搜索尺度。
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);

    //！Step4：对扩展patch进行仿射变换
    warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
    ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_);

    //！ Step5:获取去掉边界的扩展patch，注意这里求出来的patch块是在参考图像上的
    //！ Question: 为什么要先做扩展patch块的仿射变换，然后剔除patch呢，为什么不一次性做patch的？？？
    createPatchFromPatchWithBorder();

    //! 对特征点进行尺度变换
    // px_cur should be set
    Vector2d px_scaled(px_cur/(1<<search_level_));

    //! Step6：用逆向构造法，对图像特征进行对齐，得到优化之后的Feature的位置
    //! 按照角点像素和边缘特征分开处理
    //! Feature在构造的时候就构造为CORNER
    bool success = false;
    //! 处理边缘特征
    if(ref_ftr_->type == Feature::EDGELET)
    {
        Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
        dir_cur.normalize();
        success = feature_alignment::align1D(
        cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
        patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
    }
    else
    {
        success = feature_alignment::align2D(
        cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
        options_.align_max_iter, px_scaled);
    }

    //! 恢复到原尺度下
    px_cur = px_scaled * (1<<search_level_);
    return success;
}

/**
 * 极线搜索寻找匹配
 * @param ref_frame     参考帧
 * @param cur_frame     当前帧
 * @param ref_ftr       要匹配的特征点
 * @param d_estimate    上一次估计的逆深度的均值
 * @param d_min         本次投影的最小逆深度
 * @param d_max         本次投影的最大逆深度
 * @param depth         新的深度测量值
 * @return
 */
bool Matcher::findEpipolarMatchDirect(const Frame& ref_frame, const Frame& cur_frame,
                                      const Feature& ref_ftr, const double d_estimate,
                                      const double d_min, const double d_max, double& depth)
{
    //! 求取参考帧与当前帧之间的位姿变换
    SE3 T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
    int zmssd_best = PatchScore::threshold();
    Vector2d uv_best;

    //! Step1：求取参考帧上的极线
    // Compute start and end of epipolar line in old_kf for match search, on unit plane!
    Vector2d A = vk::project2d(T_cur_ref * (ref_ftr.f*d_min));
    Vector2d B = vk::project2d(T_cur_ref * (ref_ftr.f*d_max));
    epi_dir_ = A - B;

    //! Step2：计算仿射变换矩阵
    // Compute affine warp matrix
    warp::getWarpMatrixAffine(*ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
                              d_estimate, T_cur_ref, ref_ftr.level, A_cur_ref_);

    //! Step3：进行特征点的预选
    // feature pre-selection
    reject_ = false;
    if(ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering)
    {
        //! 对特征点的梯度做仿射变换
        const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
        //! 求取极线与特征点梯度方向的夹角
        const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
        //! 如果角度过小，则返回
        if(cosangle < options_.epi_search_edgelet_max_angle)
        {
            reject_ = true;
            return false;
        }
    }

    //! Step4：得到最佳的搜索金字塔层
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);

    //! Step5：得到当前帧上的极线长度
    // Find length of search range on epipolar line
    Vector2d px_A(cur_frame.cam_->world2cam(A));
    Vector2d px_B(cur_frame.cam_->world2cam(B));
    epi_length_ = (px_A-px_B).norm() / (1<<search_level_);

    //! Step6：在参考帧上对扩展patch做仿射变换
    // Warp reference patch at ref_level
    warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                     ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_);
    //！获取去掉边界的扩展patch
    createPatchFromPatchWithBorder();

    //! 如果极线长度小于两个像素点
    if(epi_length_ < 2.0)
    {
        //! 获取极线的中点
        px_cur_ = (px_A+px_B)/2.0;
        Vector2d px_scaled(px_cur_/(1<<search_level_));
        bool res;
        if(options_.align_1d)
            res = feature_alignment::align1D(cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
                                             patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
        else
            //! 逆向法对齐
            res = feature_alignment::align2D(cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
                                             options_.align_max_iter, px_scaled);
        if(res)
        {
            px_cur_ = px_scaled*(1<<search_level_);
            if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
                return true;
        }
        return false;
    }

    //! 每次搜索一个像素点
    size_t n_steps = epi_length_/0.7; // one step per pixel
    //! 每一次搜索的步长
    Vector2d step = epi_dir_/n_steps;

    //! 如果大于最大步长，说明极线求取有错，直接返回
    if(n_steps > options_.max_epi_search_steps)
    {
        printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
        n_steps, epi_length_, d_min, d_max);
        return false;
    }

    // for matching, precompute sum and sum2 of warped reference patch
    int pixel_sum = 0;
    int pixel_sum_square = 0;
    PatchScore patch_score(patch_);

    // now we sample along the epipolar line
    Vector2d uv = B-step;
    Vector2i last_checked_pxi(0,0);
    ++n_steps;
    //! Step7：沿着极线进行匹配
    for(size_t i=0; i<n_steps; ++i, uv+=step)
    {
        //! Step7.1：将点投影到相机平面，然后做一次四舍五入
        Vector2d px(cur_frame.cam_->world2cam(uv));
        Vector2i pxi(px[0]/(1<<search_level_)+0.5,
        px[1]/(1<<search_level_)+0.5); // +0.5 to round to closest int

        if(pxi == last_checked_pxi)
            continue;
        last_checked_pxi = pxi;

        //! 检查patch块是否在图像之内
        // check if the patch is full within the new frame
        if(!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_))
            continue;

        //! 在当前帧上构造patch块，获得patch块头的地址
        // TODO interpolation would probably be a good idea
        uint8_t* cur_patch_ptr = cur_frame.img_pyr_[search_level_].data
                                 + (pxi[1]-halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
                                 + (pxi[0]-halfpatch_size_);

        int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);

        if(zmssd < zmssd_best)
        {
            zmssd_best = zmssd;
            uv_best = uv;
        }
    }

    //! 对最佳匹配点三角化得到深度
    if(zmssd_best < PatchScore::threshold())
    {
        //! 是否需要再做一次对齐
        if(options_.subpix_refinement)
        {
            px_cur_ = cur_frame.cam_->world2cam(uv_best);
            Vector2d px_scaled(px_cur_/(1<<search_level_));
            bool res;
            if(options_.align_1d)
                res = feature_alignment::align1D(
                cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
                patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
            else
                res = feature_alignment::align2D(
                cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
                options_.align_max_iter, px_scaled);
            if(res)
            {
                px_cur_ = px_scaled*(1<<search_level_);
                if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
                    return true;
            }
            return false;
        }
        px_cur_ = cur_frame.cam_->world2cam(uv_best);
        if(depthFromTriangulation(T_cur_ref, ref_ftr.f, vk::unproject2d(uv_best).normalized(), depth))
            return true;
    }
    return false;
}

} // namespace svo
