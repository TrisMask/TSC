#include "provider.h"

#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <algorithm>

struct Transform
{
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
};

static inline Eigen::Matrix4f to_eigen(const Transform &tr)
{
    // T = s.Matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    // Rx = s.Matrix([[1, 0, 0, 0], [0, s.cos(rx), -s.sin(rx), 0], [0, s.sin(rx), s.cos(rx), 0], [0, 0, 0, 1]])
    // Ry = s.Matrix([[s.cos(ry), 0, s.sin(ry), 0], [0, 1, 0, 0], [-s.sin(ry), 0, s.cos(ry), 0], [0, 0, 0, 1]])
    // Rz = s.Matrix([[s.cos(rz), -s.sin(rz), 0, 0], [s.sin(rz), s.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    // Transformation matrix
    // M = T * Rz * Ry * Rx

    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 3) = tr.x;
    m(1, 3) = tr.y;
    m(2, 3) = tr.z;

    auto cr = cosf(tr.roll);
    auto sr = sinf(tr.roll);
    auto cp = cosf(tr.pitch);
    auto sp = sinf(tr.pitch);
    auto cy = cosf(tr.yaw);
    auto sy = sinf(tr.yaw);

    m(0, 0) = cp * cy;
    m(0, 1) = cy * sp * sr - cr * sy;
    m(0, 2) = sr * sy + cr * cy * sp;
    m(1, 0) = cp * sy;
    m(1, 1) = cr * cy + sp * sr * sy;
    m(1, 2) = cr * sp * sy - cy * sr;
    m(2, 0) = -sp;
    m(2, 1) = cp * sr;
    m(2, 2) = cr * cp;

    return m;
}

std::vector<Eigen::Matrix4f> load_pose9(const char *pose)
{
    // time , x, y, z, qx, qy, qz, qw;
    std::vector<Eigen::Matrix4f> poses;
    FILE *f = fopen(pose, "r");
    float x, y, z, qx, qy, qz, qw;
    while (fscanf(f, "%*f %f %f %f %f %f %f %f\n", &x, &y, &z, &qx, &qy, &qz, &qw) == 7)
    {
        // do something with x, y, z, roll, pitch, yaw
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pose.block<3, 3>(0, 0) = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
        pose.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
        poses.push_back(pose);
    }
    fclose(f);
    return poses;
}

bool numberic_filename_cmp(const std::string &a, const std::string &b)
{
    std::filesystem::path pa(a);
    std::filesystem::path pb(b);

    return std::stoi(pa.stem().string()) < std::stoi(pb.stem().string());
}

struct PCDProvider : Provider
{
    std::vector<Eigen::Matrix4f> poses;
    std::vector<std::string> clouds_file;

    PCDProvider(const char *velodyne_dir, const char *poses) : poses(load_pose9(poses))
    {
        std::vector<std::string> files;
        for (const auto &entry : std::filesystem::directory_iterator(velodyne_dir))
        {
            files.push_back(entry.path().string());
        }

        std::sort(files.begin(), files.end(), numberic_filename_cmp);

        for (const auto &file : files)
        {
            clouds_file.push_back(file);
        }
    }

    Eigen::Matrix4f pose(size_t index) override
    {
        return poses[index];
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(size_t index) override
    {
        auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

        if (pcl::io::loadPCDFile(clouds_file[index], *cloud) == -1)
        {
            PCL_ERROR("Couldn't read file %s\n", clouds_file[index].c_str());
        }

        return cloud;
    }

    size_t size() override
    {
        return poses.size();
    }
};

Provider *create_provider_p(const char *name)
{
    const char *de = strstr(name, ":");
    if (de == nullptr)
    {
        return nullptr;
    }

    std::string velodyne_dir(name, de - name);
    std::string poses(de + 1);

    return new PCDProvider(velodyne_dir.c_str(), poses.c_str());
}
