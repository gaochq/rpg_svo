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

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection 
{

    AbstractDetector::AbstractDetector(
        const int img_width,
        const int img_height,
        const int cell_size,
        const int n_pyr_levels) :
    cell_size_(cell_size),
    n_pyr_levels_(n_pyr_levels),
    grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
    grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
    grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
    {}

    void AbstractDetector::resetGrid()
    {
        std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
    }

    //! 将已存在特征点的cell单元设置为占有
    void AbstractDetector::setExistingFeatures(const Features& fts)
    {
        std::for_each(fts.begin(), fts.end(), [&](Feature* i)
        {
            grid_occupancy_.at(static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
                               + static_cast<int>(i->px[0]/cell_size_)) = true;
        }
        );
    }

    //！ 将单元格设置为占有
    void AbstractDetector::setGridOccpuancy(const Vector2d& px)
    {
        grid_occupancy_.at(
            static_cast<int>(px[1]/cell_size_)*grid_n_cols_
            + static_cast<int>(px[0]/cell_size_)) = true;
    }

    FastDetector::FastDetector(
        const int img_width,
        const int img_height,
        const int cell_size,
        const int n_pyr_levels):
            AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
    {}

    /*
     * 对某一帧图像进行Fast角点检测，每层金字塔都进行角点检测
     *
     */
    void FastDetector::detect(Frame* frame, const ImgPyr& img_pyr,
                            const double detection_threshold, Features& fts)
    {
        Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));

        //! 对每层金字塔都进行fast检测
        for(int L=0; L<n_pyr_levels_; ++L)
        {
            const int scale = (1<<L);
            vector<fast::fast_xy> fast_corners;
#if __SSE2__
            //! 使用SSE指令集提取Fast角点
            fast::fast_corner_detect_10_sse2(
                (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
                img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
            fast::fast_corner_detect_9_neon(
                (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
                img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
            fast::fast_corner_detect_10(
                (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
                img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif
            vector<int> scores, nm_corners;
            fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
            fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

            for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
            {
                fast::fast_xy& xy = fast_corners.at(*it);
                const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                            + static_cast<int>((xy.x*scale)/cell_size_);

                //! 如果格子里有特征，则直接返回即可
                if(grid_occupancy_[k])
                    continue;

                //! 利用shi-Tomasi算法对检测出的角点进行进一步的确定
                const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
                if(score > corners.at(k).score)
                    corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
            }
        }

        // Create feature for every corner that has high enough corner score
        // 将得分大于阈值的角点当做特征点返回
        std::for_each(corners.begin(), corners.end(), [&](Corner& c)
        {
            if(c.score > detection_threshold)
                fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
        }
        );

        resetGrid();
    }

} // namespace feature_detection
} // namespace svo

/*
 float FastDetector::shiTomasiScore(const cv::Mat& img, int u, int v)
	{
		assert(img.type() == CV_8UC1);

		float dXX = 0.0;
		float dYY = 0.0;
		float dXY = 0.0;
		const int halfbox_size = 4;
		const int box_size = 2 * halfbox_size;
		const int box_area = box_size*box_size;
		const int x_min = u - halfbox_size;
		const int x_max = u + halfbox_size;
		const int y_min = v - halfbox_size;
		const int y_max = v + halfbox_size;

		if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
			return 0.0; // 面片太靠近边界，返回0

		const int stride = img.step.p[0];//一行元素的个数
		for (int y = y_min; y < y_max; ++y)
		{
			const uint8_t* ptr_left = img.data + stride*y + x_min - 1;
			const uint8_t* ptr_right = img.data + stride*y + x_min + 1;
			const uint8_t* ptr_top = img.data + stride*(y - 1) + x_min;
			const uint8_t* ptr_bottom = img.data + stride*(y + 1) + x_min;
			for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
			{
				float dx = *ptr_right - *ptr_left;
				float dy = *ptr_bottom - *ptr_top;
				dXX += dx*dx;
				dYY += dy*dy;
				dXY += dx*dy;
			}
		}

		// 返回小的特征值
		dXX = dXX / (2.0 * box_area);
		dYY = dYY / (2.0 * box_area);
		dXY = dXY / (2.0 * box_area);
		return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
	}
*/
