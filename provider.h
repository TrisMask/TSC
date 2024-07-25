#ifndef __provider_h__
#define __provider_h__

#include <vector>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct Provider
{
    virtual Eigen::Matrix4f pose(size_t index) = 0;
    virtual pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(size_t index) = 0;

    virtual size_t size() = 0;
};

Provider *create_provider_k(const char *name);
Provider *create_provider_p(const char *name);

#endif