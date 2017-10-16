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
#include <stdexcept>
#include <svo/reprojector.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/map.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vikit/abstract_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>

namespace svo {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map) :
    map_(map)
{
    initializeGrid(cam);
}

Reprojector::~Reprojector()
{
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
}

/**
 * 初始化图像网格
 * @param cam
 */
void Reprojector::initializeGrid(vk::AbstractCamera* cam)
{
    //! 设定网格的数目和大小
    grid_.cell_size = Config::gridSize();
    grid_.grid_n_cols = ceil(static_cast<double>(cam->width())/grid_.cell_size);
    grid_.grid_n_rows = ceil(static_cast<double>(cam->height())/grid_.cell_size);
    grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);

    //！ 在Grid中创建cell
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
    grid_.cell_order.resize(grid_.cells.size());
    for(size_t i=0; i<grid_.cells.size(); ++i)
        grid_.cell_order[i] = i;

    //! 将cell的顺序，随机排列
    random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
}

void Reprojector::resetGrid()
{
    n_matches_ = 0;
    n_trials_ = 0;
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
}

/**
 *  挑选与当前帧有共视关系的关键帧，以及共视的mappoint的个数
 *  并寻找这些mappoint在本帧图像上的最佳的投影点，具体就是将本帧图像分为多个格子，然后每个格子只有一个投影点
 *  在寻找最佳投影点的时候使用逆向构造法，使得光度误差最小
 * @param frame
 * @param overlap_kfs <共视帧， 共视程度>
 */
void Reprojector::reprojectMap(FramePtr frame, std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
{
    //! Step1：将格子复位
    resetGrid();

    //！Step2：挑选与当前帧有共视关系的关键帧
    // Identify those Keyframes which share a common field of view.
    SVO_START_TIMER("reproject_kfs");
    list< pair<FramePtr,double> > close_kfs;
    map_.getCloseKeyframes(frame, close_kfs);

    //! Step3：按照两帧距离的远近进行排序
    // Sort KFs with overlap according to their closeness
    close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
    boost::bind(&std::pair<FramePtr, double>::second, _2));

    //！对所有重叠部分的N个关键帧对应的mappoints进行重投影，我们只存储格子中特征点减少的
    // Reproject all mappoints of the closest N kfs with overlap. We only store
    // in which grid cell the points fall.

    //! 只提取最近的10个关键帧
    size_t n = 0;
    overlap_kfs.reserve(options_.max_n_kfs);

    //! Step4：在close_kfs中按照最近的10个帧遍历, 并添加overlap_kfs的第二个参数，即mappoint的个数
    for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end(); it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame, ++n)
    {
        //！ Step4.1：加入overlap_kfs的第一个参数(共视帧)
        FramePtr ref_frame = it_frame->first;
        overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));

        //! Step4.2：遍历共视帧的每一个Feature，然后叠加两帧共视的Mappoint个数，将其作为共视程度
        // Try to reproject each mappoint that the other KF observes
        for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end(); it_ftr!=ite_ftr; ++it_ftr)
        {
            // check if the feature has a mappoint assigned
            if((*it_ftr)->point == NULL)
                continue;

            // make sure we project a point only once
            if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
                continue;
            (*it_ftr)->point->last_projected_kf_id_ = frame->id_;

            //! 增加共视的Mappoint的个数
            if(reprojectPoint(frame, (*it_ftr)->point))
                overlap_kfs.back().second++;
        }
    }
    SVO_STOP_TIMER("reproject_kfs");

    // Now project all point candidates
    //! Step5：剔除不符合条件的候选mappoint
    SVO_START_TIMER("reproject_candidates");
    {
        boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
        auto it=map_.point_candidates_.candidates_.begin();
        while(it!=map_.point_candidates_.candidates_.end())
        {
            //! 剔除不符合条件的候选点(投影后不在当前图像帧上)
            if(!reprojectPoint(frame, it->first))
            {
                it->first->n_failed_reproj_ += 3;
                if(it->first->n_failed_reproj_ > 30)
                {
                    map_.point_candidates_.deleteCandidate(*it);
                    it = map_.point_candidates_.candidates_.erase(it);
                    continue;
                }
            }
            ++it;
        }
    } // unlock the mutex when out of scope
    SVO_STOP_TIMER("reproject_candidates");

    // Now we go through each grid cell and select one point to match.
    // At the end, we should have at maximum one reprojected point per cell.
    //! Step6: 遍历每一个Gridcell。通过投影之后，每一个格子会有多个投影点，最终只选择一个质量最好的。
    SVO_START_TIMER("feature_align");
    for(size_t i=0; i<grid_.cells.size(); ++i)
    {
        // we prefer good quality points over unkown quality (more likely to match)
        // and unknown quality over candidates (position not optimized)
        //! cell_order: vector<int>
        if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
            ++n_matches_;

        //! 如果匹配的特征点个数大于最大跟踪特征点个数
        if(n_matches_ > (size_t) Config::maxFts())
            break;
    }
    SVO_STOP_TIMER("feature_align");
}

//! 将cell按照所包含feature的质量排序
bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
    if(lhs.pt->type_ > rhs.pt->type_)
        return true;
    return false;
}

/**
 * 经过reprojectMap，每个格子会有很多Features，然后在每个格子中只选择一个最好的添加到当前帧
 * 遍历所有cell，将其投影到当前帧上
 * @param cell
 * @param frame
 * @return
 */
bool Reprojector::reprojectCell(Cell& cell, FramePtr frame)
{
    //! Cell: std::list<Candidate>  Candidate: 3D点和对应的特征点
    //！ Step1：将cell按照3D点的质量好坏排序
    cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));

    //！Step2： 遍历cell中的Candidate
    Cell::iterator it=cell.begin();
    while(it!=cell.end())
    {
        //! 重投影单元格的次数
        ++n_trials_;

        //！ Step2.1: 检查候选点的类型，不合适则剔除
        if(it->pt->type_ == Point::TYPE_DELETED)
        {
            it = cell.erase(it);
            continue;
        }

        //! Step2.2: 通过逆向构造的方法对patch块对齐，进而得到准确的Feature位置(px)
        bool found_match = true;
        if(options_.find_match_direct)
            found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);

        //! Step2.3: 如果3D点长期找不到，则将其剔除即可
        if(!found_match)
        {
            it->pt->n_failed_reproj_++;
            if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
                map_.safeDeletePoint(it->pt);
            if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
                map_.point_candidates_.deleteCandidatePoint(it->pt);
            it = cell.erase(it);
            continue;
        }

        //! Step2.4: 根据3D点成功投影的次数，设定其属性。
        it->pt->n_succeeded_reproj_++;
        if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
            it->pt->type_ = Point::TYPE_GOOD;

        //! Step2.4: 构造新的Feature，并将其添加到当前帧中
        Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
        frame->addFeature(new_feature);

        // Here we add a reference in the feature to the 3D point, the other way
        // round is only done if this frame is selected as keyframe.
        //！添加Feature对应的3D点
        new_feature->point = it->pt;

        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
            new_feature->type = Feature::EDGELET;
            new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
            new_feature->grad.normalize();
        }

        // If the keyframe is selected and we reproject the rest, we don't have to
        // check this point anymore.
        it = cell.erase(it);

        // Maximum one point per cell.
        return true;
    }
    return false;
}

//! 重投影3D点，将投影在帧内的像素点，作为候选特征，往grid中存投影成功的候选点
bool Reprojector::reprojectPoint(FramePtr frame, Point* point)
{
    Vector2d px(frame->w2c(point->pos_));
    if(frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
    {
        //! 获取特征点对应的格子index
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols +
                      static_cast<int>(px[0]/grid_.cell_size);

        grid_.cells.at(k)->push_back(Candidate(point, px));
        return true;
    }
    return false;
}

} // namespace svo
