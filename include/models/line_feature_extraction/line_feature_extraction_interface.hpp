/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified: 2022-04-28 14:43
#
# Filename: line_feature_extraction_interface.hpp
#
# Description: interface of the line feature extraction.
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_LINE_FEATURE_EXTRACTION_INTERFACE_HPP_
#define VSION_LOCALIZATION_MODELS_LINE_FEATURE_EXTRACTION_INTERFACE_HPP_

#include "sensor_data/cloud_data.hpp"
#include "sensor_data/line_feature.hpp"
#include "tools/convert_matrix.hpp"
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <yaml-cpp/yaml.h>

namespace vision_localization
{
class LineFeatureExtractionInterface
{
public:
  virtual ~LineFeatureExtractionInterface() = default;

  virtual bool Extract(const CloudData::CLOUD_PTR &input_cloud, const int semantic_id, std::vector<LineFeature> &lines,
                       std::vector<int> &indices) = 0;
  virtual bool Extract(const pcl::PointCloud<pcl::PointXY>::Ptr &input_cloud, const int semantic_id,
                       std::vector<LineFeature2D> &lines, std::vector<int> &indices) = 0;

  // debug
  CloudData2D cloud_2d;

protected:
  // CloudData::CLOUD_PTR input_source_;
};
}  // namespace vision_localization

#endif
