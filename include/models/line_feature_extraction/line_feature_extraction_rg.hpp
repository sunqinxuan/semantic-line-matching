/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified: 2022-04-30 09:59
#
# Filename: line_feature_extraction_rg.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_LINE_FEATURE_EXTRACTION_RG_HPP_
#define VSION_LOCALIZATION_MODELS_LINE_FEATURE_EXTRACTION_RG_HPP_

#include "global_defination/message_print.hpp"
#include "models/line_feature_extraction/line_feature_extraction_interface.hpp"
#include <Eigen/SVD>
#include <deque>
#include <fstream>
#include <pcl/kdtree/kdtree_flann.h>

namespace vision_localization
{
class Region
{
public:
  Region()
  {
  }

  std::size_t &at(std::size_t i)
  {
    return indices_.at(i);
  }
  const std::size_t &at(std::size_t i) const
  {
    return indices_.at(i);
  }
  const std::size_t &operator[](std::size_t i) const
  {
    return indices_[i];
  }
  std::size_t size()
  {
    return indices_.size();
  }
  void clear()
  {
    indices_.clear();
    region_centroid_.setZero();
    region_covariance_.setZero();
  }
  std::vector<std::size_t>::iterator begin()
  {
    return indices_.begin();
  }
  std::vector<std::size_t>::iterator end()
  {
    return indices_.end();
  }

  const Eigen::Vector2f GetCentroid() const
  {
    return region_centroid_;
  }

  const Eigen::Vector2f GetDirection() const
  {
    return region_direction_;
  }

  const float GetLength() const
  {
    return length_;
  }

  void AddPoint(const CloudData2D &cloud, std::size_t idx);
  void ToLineFeature(const CloudData2D &cloud, const int semantic_id, LineFeature &line);
  void ToLineFeature2D(const CloudData2D &cloud, const int semantic_id, LineFeature2D &line);

  const float GetResidual() const
  {
    return residual_;
  }
  void ComputeResidual(const CloudData2D &cloud)
  {
    residual_ = 0.0;
    for (std::size_t i = 0; i < indices_.size(); i++) {
      std::size_t idx = indices_[i];
      CloudData2D::POINTXYI pt = cloud.at(idx);
      Eigen::Vector2f p(pt.x, pt.y);
      Eigen::Vector2f d(region_direction_(1), -region_direction_(0));
      residual_ += fabs(d.dot(p - region_centroid_));
      // residual_ += region_direction_.cross(p - region_centroid_).norm();
    }
    residual_ /= float(indices_.size());
  }

private:
  std::vector<std::size_t> indices_;

  Eigen::Matrix2f region_covariance_ = Eigen::Matrix2f::Zero();
  Eigen::Vector2f region_centroid_ = Eigen::Vector2f::Zero();
  Eigen::Vector2f region_direction_ = Eigen::Vector2f::Zero();

  float length_ = 0.0;
  float residual_ = 0.0;
};

class LineFeatureExtractionRG : public LineFeatureExtractionInterface
{
public:
  using VecBins = std::vector<std::deque<std::size_t>>;
  using VecRegion = std::vector<Region>;

  LineFeatureExtractionRG(const YAML::Node &node);
  bool Extract(const CloudData::CLOUD_PTR &input_cloud, const int semantic_id, std::vector<LineFeature> &lines,
               std::vector<int> &indices) override;
  bool Extract(const pcl::PointCloud<pcl::PointXY>::Ptr &input_cloud, const int semantic_id,
               std::vector<LineFeature2D> &lines, std::vector<int> &indices) override;

private:
  // cloud_3d could contain several semantic labels;
  bool Projection3D2D(const CloudData::CLOUD_PTR &cloud_3d, const int label_id, CloudData2D &cloud);

  // points in cloud_3d must be of the same semantic label;
  bool Projection3D2D(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud_3d, const int label_id, CloudData2D &cloud);

  bool ComputeLocalLinearity(CloudData2D &cloud);
  bool PseudoOrdering(CloudData2D &cloud, VecBins &pseudo_ordering_bins);
  bool RegionSegmentation(CloudData2D &cloud, VecBins &pseudo_ordering_bins, VecRegion &regions);
  bool RegionGrow(CloudData2D &cloud, std::size_t seed_idx, Region &region);
  float AngleDiff(const Eigen::Vector2f &dir1, const Eigen::Vector2f &dir2);

private:
  float tmp_radius_neighbors_svd_ = 0.3;
  float tmp_local_direction_threshold_ = 0.52;  // 30deg
  int tmp_region_size_threshold_ = 20;

  // int num_neighbors_svd_ = 20;
  float radius_neighbors_svd_ = 0.3;
  float local_direction_threshold_ = 0.52;  // 30deg
  int region_size_threshold_ = 20;
  float region_length_threshold_ = 0.2;
  float region_residual_threshold_ = 0.02;

  CloudData2D semantic_clouds_2d_;
  VecBins vec_pseudo_ordering_bins_;
  VecRegion vec_semantic_regions_;

  // pseudo_ordering_bins_
  // [0.0,0.1),[0.1,0.2),...,[0.9,1.0];
  // std::vector<std::deque<std::size_t>> pseudo_ordering_bins_;

  // std::vector<Region> cloud_2d_regions_;
};
}  // namespace vision_localization

#endif
