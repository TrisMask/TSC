#include <filesystem>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

const char *kitti_sequences_root = "/home/jlurobot/桌面/sequences";

auto load_kitti_calib(std::filesystem::path seq_root)
{
    auto calib = seq_root / "calib.txt";

    std::ifstream calib_file(calib);

    // ignore first 4 lines
    for (int i = 0; i < 4; i++)
    {
        std::string line;
        std::getline(calib_file, line);
    }

    // Tr: 3x4 matrix
    Eigen::Matrix4f Tr;

    calib_file.ignore(4); // ignore "Tr: "
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            calib_file >> Tr(i, j);
        }
    }

    Tr(3, 0) = 0.0f;
    Tr(3, 1) = 0.0f;
    Tr(3, 2) = 0.0f;
    Tr(3, 3) = 1.0f;

    return Tr;
}

std::vector<Eigen::Matrix4f> load_kitti_gt(std::filesystem::path seq_root)
{
    auto gt = seq_root / "poses.txt";

    FILE *fp = fopen(gt.c_str(), "r");
    std::vector<Eigen::Matrix4f> poses;
    float data[12];
    while (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f", data, data + 1, data + 2, data + 3, data + 4, data + 5, data + 6, data + 7, data + 8, data + 9, data + 10, data + 11) == 12)
    {
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

        for (int i = 0; i < 12; i++)
        {
            pose(i / 4, i % 4) = data[i];
        }

        pose(3, 0) = 0.0f;
        pose(3, 1) = 0.0f;
        pose(3, 2) = 0.0f;
        pose(3, 3) = 1.0f;
        poses.push_back(pose);
    }

    fclose(fp);

    return poses;
}

std::vector<Eigen::Matrix4f> load_kitti_pose(std::filesystem::path seq_root)
{
    auto poses = load_kitti_gt(seq_root);
    auto Tr = load_kitti_calib(seq_root);

    auto Tr_inv = Tr.inverse();

    for (auto &pose : poses)
    {
        pose = Tr_inv * pose * Tr;
    }

    return poses;
}

#include "provider.h"

struct KittiProvider : Provider
{
    std::vector<Eigen::Matrix4f> poses;

    std::filesystem::path scans_root;

    KittiProvider(std::filesystem::path seq_root)
    {
        poses = load_kitti_pose(seq_root);

        scans_root = seq_root / "velodyne";
    }

    virtual Eigen::Matrix4f pose(size_t index)
    {
        return poses[index];
    }

    virtual pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(size_t index)
    {
        char scan_name[16];
        snprintf(scan_name, 16, "%06zd.bin", index);

        auto scan_path = scans_root / scan_name;

        FILE *fp = fopen(scan_path.c_str(), "rb");

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp) / sizeof(float);
        fseek(fp, 0, SEEK_SET);

        float *data = new float[size / sizeof(float)];

        fread(data, size, 1, fp);

        fclose(fp);

        cloud->resize(size / 16);

        for (size_t i = 0; i < size / 16; i++)
        {
            cloud->points[i].x = data[i * 4];
            cloud->points[i].y = data[i * 4 + 1];
            cloud->points[i].z = data[i * 4 + 2];
            cloud->points[i].intensity = data[i * 4 + 3];
        }

        delete[] data;

        return cloud;
    }

    virtual size_t size()
    {
        return poses.size();
    }
};

Provider *create_provider_k(const char *name)
{
    char seq_name[256];
    snprintf(seq_name, 256, "%s/%s", kitti_sequences_root, name);
    return new KittiProvider(seq_name);
}
