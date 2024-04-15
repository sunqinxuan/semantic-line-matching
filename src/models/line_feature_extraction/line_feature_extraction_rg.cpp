/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified: 2022-04-30 10:01
#
# Filename: line_feature_extraction_rg.cpp
#
# Description:
#
************************************************/

#include "models/line_feature_extraction/line_feature_extraction_rg.hpp"

namespace vision_localization
{
void Region::AddPoint(const CloudData2D &cloud, std::size_t idx)
{
  indices_.push_back(idx);
  const CloudData2D::POINTXYI &pt = cloud.at(idx);
  Eigen::Vector2f point(pt.x, pt.y);

  std::size_t N = indices_.size();
  if (N == 1) {
    region_centroid_ = point;
    region_covariance_.setZero();
    region_direction_ = cloud.LocalLinearDirection(idx);
  } else {
    float ratio = float(N - 1) / float(N);
    Eigen::Vector2f tmp = point - region_centroid_;
    region_centroid_ = ratio * (region_centroid_ + point / float(N - 1));
    region_covariance_ = ratio * (region_covariance_ + tmp * tmp.transpose() / float(N));

    Eigen::JacobiSVD<Eigen::Matrix2f> svd(region_covariance_, Eigen::ComputeFullU);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Vector2f D = svd.singularValues();
    if (D(1) / D(0) < 0.1) {
      region_direction_ = U.block<2, 1>(0, 0);
    } else {
      region_direction_ = ratio * (region_direction_ + cloud.LocalLinearDirection(idx) / float(N - 1));
    }

    float length_tmp = 2.0 * fabs(region_direction_.dot(point - region_centroid_));
    if (length_tmp > length_) length_ = length_tmp;
  }
}

void Region::ToLineFeature(const CloudData2D &cloud, const int semantic_id, LineFeature &line)
{
  line.semantic_id = semantic_id;
  line.size = indices_.size();
  // TODO
  line.centroid = Eigen::Vector3f(region_centroid_(0), region_centroid_(1), -0.11);
  line.direction = Eigen::Vector3f(region_direction_(0), region_direction_(1), 0.0);

  float max_dist = -1.0;
  Eigen::Vector2f rel_endpoint;
  for (std::size_t i = 0; i < indices_.size(); i++) {
    std::size_t idx = indices_[i];
    CloudData2D::POINTXYI pt = cloud.at(idx);
    Eigen::Vector2f point(pt.x, pt.y);
    Eigen::Vector2f rel_proj_point = region_direction_.dot(point - region_centroid_) * region_direction_;
    if (rel_proj_point.norm() > max_dist) {
      max_dist = rel_proj_point.norm();
      rel_endpoint = rel_proj_point;
    }
  }
  Eigen::Vector3f rel_endpoint_3d(rel_endpoint(0), rel_endpoint(1), 0.0);  // TODO

  line.endpoint_1 = line.centroid + rel_endpoint_3d;
  line.endpoint_2 = line.centroid - rel_endpoint_3d;

  line.residual = residual_;
}

void Region::ToLineFeature2D(const CloudData2D &cloud, const int semantic_id, LineFeature2D &line)
{
  line.semantic_id = semantic_id;
  line.size = indices_.size();

  line.centroid = region_centroid_;
  line.direction = region_direction_;

  float max_dist = -1.0;
  Eigen::Vector2f rel_endpoint;
  for (std::size_t i = 0; i < indices_.size(); i++) {
    std::size_t idx = indices_[i];
    CloudData2D::POINTXYI pt = cloud.at(idx);
    Eigen::Vector2f point(pt.x, pt.y);
    Eigen::Vector2f rel_proj_point = region_direction_.dot(point - region_centroid_) * region_direction_;
    if (rel_proj_point.norm() > max_dist) {
      max_dist = rel_proj_point.norm();
      rel_endpoint = rel_proj_point;
    }
  }

  line.endpoint_1 = line.centroid + rel_endpoint;
  line.endpoint_2 = line.centroid - rel_endpoint;
}


LineFeatureExtractionRG::LineFeatureExtractionRG(const YAML::Node &node)
{
  // num_neighbors_svd_ = node["num_neighbors_svd"].as<int>();
  radius_neighbors_svd_ = node["radius_neighbors_svd"].as<float>();
  local_direction_threshold_ = node["local_direction_threshold"].as<float>() * M_PI / 180.0;
  region_size_threshold_ = node["region_size_threshold"].as<std::size_t>();
  region_residual_threshold_ = node["region_residual_threshold"].as<float>();
  region_length_threshold_ = node["region_length_threshold"].as<float>();
}

bool LineFeatureExtractionRG::Extract(const CloudData::CLOUD_PTR &input_cloud, const int semantic_id,
                                      std::vector<LineFeature> &lines, std::vector<int> &indices)
{
  // if (semantic_id == 1) {
  //  // no parking zone;
  //  radius_neighbors_svd_ = 0.3;
  //  local_direction_threshold_ = 20.0 * M_PI / 180.0;
  //  region_size_threshold_ = 7;
  //} else if (semantic_id == 6) {
  //  // guiding line;
  //  radius_neighbors_svd_ = 0.3;
  //  local_direction_threshold_ = 20.0 * M_PI / 180.0;
  //  region_size_threshold_ = 40;
  //} else {
  //  radius_neighbors_svd_ = tmp_radius_neighbors_svd_;
  //  local_direction_threshold_ = tmp_local_direction_threshold_;
  //  region_size_threshold_ = tmp_region_size_threshold_;
  //}
  //
  // std::vector<int> labels;
  // labels.push_back(2);  // parking lot
  // labels.push_back(4);  // lane
  // labels.push_back(5);  // lane center line

  // const std::size_t N = labels.size();
  // semantic_clouds_2d_.resize(N);
  // vec_pseudo_ordering_bins_.resize(N);
  // vec_semantic_regions_.resize(N);

  // cloud_2d.clear();
  lines.clear();
  indices.resize(input_cloud->size(), -1);
  // for (std::size_t i = 0; i < labels.size(); i++) {
  Projection3D2D(input_cloud, semantic_id, semantic_clouds_2d_);
  if (semantic_clouds_2d_.size() < 10) return false;

  ComputeLocalLinearity(semantic_clouds_2d_);
  PseudoOrdering(semantic_clouds_2d_, vec_pseudo_ordering_bins_);
  RegionSegmentation(semantic_clouds_2d_, vec_pseudo_ordering_bins_, vec_semantic_regions_);

  // DEBUG
  // for (std::size_t j = 0; j < semantic_clouds_2d_.size(); j++) {
  //  CloudData2D::POINTXYI &pt = semantic_clouds_2d_.at(j);
  //  if (!semantic_clouds_2d_.IsUsed(j)) pt.intensity = -1;
  //}
  for (std::size_t r = 0; r < vec_semantic_regions_.size(); r++) {
    Region &region = vec_semantic_regions_[r];
    region.ComputeResidual(semantic_clouds_2d_);
    // if (semantic_id == 1 || semantic_id == 6) {
    if (region.GetResidual() > region_residual_threshold_) continue;
    //}
    LineFeature line;
    region.ToLineFeature(semantic_clouds_2d_, semantic_id, line);
    lines.push_back(line);
    // Eigen::Vector2f c = region.GetCentroid();
    // Eigen::Vector2f d = region.GetDirection();
    for (std::size_t k = 0; k < region.size(); k++) {
      std::size_t idx = region.at(k);
      indices[idx] = lines.size() - 1;
      //  CloudData2D::POINTXYI &ptt = semantic_clouds_2d_.at(idx);
      //  ptt.intensity = r + 1;
      // Eigen::Vector2f p(ptt.x, ptt.y);
      // Eigen::Vector2f proj_p = d.dot(p - c) * d + c;
      // ptt.x = proj_p(0);
      // ptt.y = proj_p(1);
    }
  }
  // DEBUG
  // for (std::size_t j = 0; j < semantic_clouds_2d_.size(); j++) {
  // CloudData2D::POINTXYI &pt = semantic_clouds_2d_.at(j);
  // cloud_2d.AddPoint(pt.x, pt.y, pt.intensity);
  //}
  //}

  return true;
}

bool LineFeatureExtractionRG::Extract(const pcl::PointCloud<pcl::PointXY>::Ptr &input_cloud, const int semantic_id,
                                      std::vector<LineFeature2D> &lines, std::vector<int> &indices)
{
  // if (semantic_id == 1) {
  //  // no parking zone;
  //  radius_neighbors_svd_ = 0.3;
  //  local_direction_threshold_ = 20.0 * M_PI / 180.0;
  //  region_size_threshold_ = 7;
  //} else if (semantic_id == 6) {
  //  // guiding line;
  //  radius_neighbors_svd_ = 0.3;
  //  local_direction_threshold_ = 20.0 * M_PI / 180.0;
  //  region_size_threshold_ = 40;
  //} else {
  //  radius_neighbors_svd_ = tmp_radius_neighbors_svd_;
  //  local_direction_threshold_ = tmp_local_direction_threshold_;
  //  region_size_threshold_ = tmp_region_size_threshold_;
  //}
  // std::vector<int> labels;
  // labels.push_back(2);  // parking lot
  // labels.push_back(4);  // lane
  // labels.push_back(5);  // lane center line

  // const std::size_t N = labels.size();
  // semantic_clouds_2d_.resize(N);
  // vec_pseudo_ordering_bins_.resize(N);
  // vec_semantic_regions_.resize(N);

  // cloud_2d.clear();
  lines.clear();
  // for (std::size_t i = 0; i < labels.size(); i++) {
  if (!Projection3D2D(input_cloud, semantic_id, semantic_clouds_2d_)) return false;
  // if (semantic_clouds_2d_.size() < 10) return false;
  indices.resize(input_cloud->size(), -1);

  ComputeLocalLinearity(semantic_clouds_2d_);
  PseudoOrdering(semantic_clouds_2d_, vec_pseudo_ordering_bins_);
  RegionSegmentation(semantic_clouds_2d_, vec_pseudo_ordering_bins_, vec_semantic_regions_);

  // DEBUG
  // for (std::size_t j = 0; j < semantic_clouds_2d_.size(); j++) {
  //  CloudData2D::POINTXYI &pt = semantic_clouds_2d_.at(j);
  //  if (!semantic_clouds_2d_.IsUsed(j)) pt.intensity = -1;
  //}
  for (std::size_t r = 0; r < vec_semantic_regions_.size(); r++) {
    Region &region = vec_semantic_regions_[r];
    region.ComputeResidual(semantic_clouds_2d_);
    // if (semantic_id == 1 || semantic_id == 6) {
    if (region.GetResidual() > region_residual_threshold_) continue;
    //}
    LineFeature2D line;
    region.ToLineFeature2D(semantic_clouds_2d_, semantic_id, line);
    lines.push_back(line);
    // Eigen::Vector2f c = region.GetCentroid();
    // Eigen::Vector2f d = region.GetDirection();
    for (std::size_t k = 0; k < region.size(); k++) {
      std::size_t idx = region.at(k);
      indices[idx] = lines.size() - 1;
      //  CloudData2D::POINTXYI &ptt = semantic_clouds_2d_.at(idx);
      //  ptt.intensity = r + 1;
      // Eigen::Vector2f p(ptt.x, ptt.y);
      // Eigen::Vector2f proj_p = d.dot(p - c) * d + c;
      // ptt.x = proj_p(0);
      // ptt.y = proj_p(1);
    }
  }
  // DEBUG
  // for (std::size_t j = 0; j < semantic_clouds_2d_.size(); j++) {
  // CloudData2D::POINTXYI &pt = semantic_clouds_2d_.at(j);
  // cloud_2d.AddPoint(pt.x, pt.y, pt.intensity);
  //}
  //}

  return true;
}

bool LineFeatureExtractionRG::Projection3D2D(const CloudData::CLOUD_PTR &cloud_3d, const int label_id,
                                             CloudData2D &cloud)
{
  if (cloud_3d->size() < 10) {
    // ERROR("[LineFeatureExtractionRG][Projection3D2D] empty input 3d cloud!");
    return false;
  }
  cloud.clear();
  // TODO
  // avm_to_imu_.linear()=I;
  // avm_to_imu_.translation().z=-0.11;
  for (auto it_pt = cloud_3d->begin(); it_pt != cloud_3d->end(); ++it_pt) {
    const CloudData::POINTXYZI &pt = *it_pt;
    if (floorf(pt.intensity) == label_id) cloud.AddPoint(pt.x, pt.y, pt.intensity);
  }
  return true;
}

bool LineFeatureExtractionRG::Projection3D2D(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud_3d, const int label_id,
                                             CloudData2D &cloud)
{
  if (cloud_3d->size() < 10) {
    ERROR("[LineFeatureExtractionRG][Projection3D2D] empty input 3d cloud!");
    return false;
  }
  cloud.clear();
  for (auto it_pt = cloud_3d->begin(); it_pt != cloud_3d->end(); ++it_pt) {
    const pcl::PointXY &pt = *it_pt;
    cloud.AddPoint(pt.x, pt.y, float(label_id));
    // if (floorf(pt.intensity) == label_id) cloud.AddPoint(pt.x, pt.y, pt.intensity);
  }
  return true;
}

bool LineFeatureExtractionRG::ComputeLocalLinearity(CloudData2D &cloud)
{
  if (cloud.size() < 10) {
    ERROR("[LineFeatureExtractionRG][LocalLinearity] empty input point cloud!");
    return false;
  }
  pcl::PointCloud<pcl::PointXY>::Ptr cloud_kdtree(new pcl::PointCloud<pcl::PointXY>);
  pcl::KdTreeFLANN<pcl::PointXY>::Ptr kdtree_ptr(new pcl::KdTreeFLANN<pcl::PointXY>);
  cloud_kdtree->resize(cloud.size());
  for (std::size_t i = 0; i < cloud.size(); i++) {
    cloud_kdtree->at(i).x = cloud.at(i).x;
    cloud_kdtree->at(i).y = cloud.at(i).y;
  }
  kdtree_ptr->setInputCloud(cloud_kdtree);

  std::vector<int> nn_indices;
  std::vector<float> nn_dist_sq;
  // nn_indices.reserve(num_neighbors_svd_);
  // nn_dist_sq.reserve(num_neighbors_svd_);
  Eigen::Vector2f mu;
  Eigen::Matrix2f cov;

  float half_radius_sq = 0.25 * radius_neighbors_svd_ * radius_neighbors_svd_;

  cloud.ResizeLocalLinearity();
  for (std::size_t i = 0; i < cloud.size(); i++) {
    if (cloud.GetNeighbors(i).size() > 0) continue;

    const pcl::PointXY &query = cloud_kdtree->at(i);
    // kdtree_ptr->nearestKSearch(query, num_neighbors_svd_, nn_indices, nn_dist_sq);

    if (kdtree_ptr->radiusSearch(query, radius_neighbors_svd_, nn_indices, nn_dist_sq) < 5) {
      cloud.SetStatusInvalid(i);
      continue;
    }

    mu.setZero();
    cov.setZero();

    for (std::size_t j = 0; j < nn_indices.size(); j++) {
      const pcl::PointXY &pt = cloud_kdtree->at(nn_indices[j]);
      mu(0) += pt.x;
      mu(1) += pt.y;
      cov(0, 0) += pt.x * pt.x;
      cov(1, 0) += pt.y * pt.x;
      cov(1, 1) += pt.y * pt.y;
    }
    mu /= static_cast<float>(nn_indices.size());
    for (int k = 0; k < 2; k++) {
      for (int l = 0; l <= k; l++) {
        cov(k, l) /= static_cast<float>(nn_indices.size());
        cov(k, l) -= mu[k] * mu[l];
        cov(l, k) = cov(k, l);
      }
    }

    Eigen::JacobiSVD<Eigen::Matrix2f> svd(cov, Eigen::ComputeFullU);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Vector2f val = svd.singularValues();  // sorted in decreasing order;

    float lin = 1.0 - val(1) / val(0);
    Eigen::Vector2f dir(U(0, 0), U(1, 0));
    cloud.AddLocalLinearity(i, lin, dir, nn_indices);

    // for the neighbors within half-radius;
    // set the same local linearity and direction;
    for (std::size_t j = 0; j < nn_indices.size(); j++) {
      if (cloud.GetStatusInvalid(nn_indices[j]) || cloud.GetNeighbors(nn_indices[j]).size() > 0) continue;
      if (nn_dist_sq[j] < half_radius_sq) {
        cloud.AddLocalLinearity(nn_indices[j], lin, dir, nn_indices);
      }
    }
  }

  return true;
}

bool LineFeatureExtractionRG::PseudoOrdering(CloudData2D &cloud, VecBins &pseudo_ordering_bins)
{
  if (cloud.size() < 10) {
    ERROR("[LineFeatureExtractionRG][PseudoOrdering] empty input point cloud!");
    return false;
  }
  pseudo_ordering_bins.resize(10);
  for (std::size_t i = 0; i < pseudo_ordering_bins.size(); i++) {
    pseudo_ordering_bins[i].clear();
  }

  for (std::size_t i = 0; i < cloud.size(); i++) {
    if (cloud.GetStatusInvalid(i)) continue;

    // int idx = floorf(cloud.LocalLinearity(i) * 10.0);
    int idx = floorf(fabs(1.0 - cloud.LocalLinearity(i)) * 10.0);
    if (idx > 9) idx = 9;
    pseudo_ordering_bins[idx].push_back(i);
  }

  return true;
}

bool LineFeatureExtractionRG::RegionSegmentation(CloudData2D &cloud, VecBins &pseudo_ordering_bins, VecRegion &regions)
{
  regions.clear();
  // for (std::size_t i = 0; i < pseudo_ordering_bins.size(); i++) {
  for (std::size_t i = 0; i < pseudo_ordering_bins.size() * 0.5; i++) {
    for (auto it = pseudo_ordering_bins[i].begin(); it != pseudo_ordering_bins[i].end(); ++it) {
      const std::size_t seed_idx = *it;

      if (cloud.IsUsed(seed_idx)) continue;
      Region reg;
      if (RegionGrow(cloud, seed_idx, reg)) regions.push_back(reg);
    }
  }

  return true;
}

bool LineFeatureExtractionRG::RegionGrow(CloudData2D &cloud, std::size_t seed_idx, Region &region)
{
  if (seed_idx < 0 || seed_idx >= cloud.size()) {
    ERROR("[LineFeatureExtractionRG][RegionGrow] wrong seed point index!");
    return false;
  }

  region.clear();
  region.AddPoint(cloud, seed_idx);

  for (std::size_t i = 0; i < region.size(); i++) {
    std::size_t idx = region[i];

    const std::vector<int> &neighbors = cloud.GetNeighbors(idx);

    for (auto itn = neighbors.begin(); itn != neighbors.end(); ++itn) {
      std::size_t idxn = *itn;
      if (cloud.GetStatusInvalid(idxn)) continue;
      if (cloud.IsUsed(idxn)) continue;
      if (cloud.LocalLinearity(idxn) < 0.5) continue;

      Eigen::Vector2f region_dir = region.GetDirection();
      Eigen::Vector2f point_dir = cloud.LocalLinearDirection(idxn);

      if (AngleDiff(region_dir, point_dir) < local_direction_threshold_) {
        region.AddPoint(cloud, idxn);
        cloud.SetStatusUsed(idxn);
      }
    }

    if (region.GetLength() > region_length_threshold_) break;
  }

  if (region.size() < region_size_threshold_) {
    for (std::size_t i = 0; i < region.size(); i++) {
      std::size_t idx = region[i];
      cloud.SetStatusUnused(idx);
    }
    return false;
  } else
    return true;
}

float LineFeatureExtractionRG::AngleDiff(const Eigen::Vector2f &dir1, const Eigen::Vector2f &dir2)
{
  return acos(fabs(dir1.dot(dir2)));
}
}  // namespace vision_localization
