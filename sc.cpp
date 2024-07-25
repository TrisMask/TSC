#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZI; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// namespace SC2
// {

void coreImportTest(void);

// sc param-independent helper functions
float xy2theta(const float &_x, const float &_y);
MatrixXd circshift(MatrixXd &_mat, int _num_shift);
std::vector<float> eig2stdvec(MatrixXd _eigmat);

class SCManager
{
public:
    SCManager() = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType> &_scan_down);
    Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd &_desc);
    Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc);

    int fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2);
    double distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2);                           // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnScanContext(MatrixXd &_sc1, MatrixXd &_sc2); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down);
    std::pair<int, float> detectLoopClosureID(size_t artop1, int *); // int: nearest node index, float: relative yaw

public:
    // hyper parameters ()
    const double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    const int PC_NUM_RING = 20;        // 20 in the original paper (IROS 18)
    const int PC_NUM_SECTOR = 60;      // 60 in the original paper (IROS 18)
    const double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    const int NUM_EXCLUDE_RECENT = 50;            // simply just keyframe gap, but node position distance-based exclusion is ok.
    const int NUM_CANDIDATES_FROM_TREE_Init = 10; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    const double SEARCH_RATIO = 0.1;   // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    // const double SC_DIST_THRES = 0.5; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config
    const int TREE_MAKING_PERIOD_ = 50; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / if you want to find a very recent revisits use small value of it (it is enough fast ~ 5-50ms wrt N.).
    int tree_making_period_conter = 0;

    // data
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

}; // SCManager

// } // namespace SC2

// namespace SC2
// {

void coreImportTest(void)
{
    cout << "scancontext lib is successfully imported." << endl;
} // coreImportTest

float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}

float xy2theta(const float &_x, const float &_y)
{
    if (_x >= 0 & _y >= 0)
        return (180 / M_PI) * atan(_y / _x);

    if (_x < 0 & _y >= 0)
        return 180 - ((180 / M_PI) * atan(_y / (-_x)));

    if (_x < 0 & _y < 0)
        return 180 + ((180 / M_PI) * atan(_y / _x));

    if (_x >= 0 & _y < 0)
        return 360 - ((180 / M_PI) * atan((-_y) / _x));

    return 0.0f;
} // xy2theta

MatrixXd circshift(MatrixXd &_mat, int _num_shift)
{
    // shift columns to right direction
    assert(_num_shift >= 0);

    if (_num_shift == 0)
    {
        MatrixXd shifted_mat(_mat);
        return shifted_mat; // Early return
    }

    MatrixXd shifted_mat = MatrixXd::Zero(_mat.rows(), _mat.cols());
    for (int col_idx = 0; col_idx < _mat.cols(); col_idx++)
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;

} // circshift

std::vector<float> eig2stdvec(MatrixXd _eigmat)
{
    std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
    return vec;
} // eig2stdvec

double SCManager::distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2)
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++)
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);

        if (col_sc1.norm() == 0 | col_sc2.norm() == 0)
            continue; // don't count this sector pair.

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

} // distDirectSC

int SCManager::fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++)
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm = vkey_diff.norm();
        if (cur_diff_norm < min_veky_diff_norm)
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey

std::pair<double, int> SCManager::distanceBtnScanContext(MatrixXd &_sc1, MatrixXd &_sc2)
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

    const int SEARCH_RADIUS = round(0.5 * SEARCH_RATIO * _sc1.cols()); // a half of search range
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++)
    {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. fast columnwise diff
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for (int num_shift : shift_idx_search_space)
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
        if (cur_sc_dist < min_sc_dist)
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

MatrixXd SCManager::makeScancontext(pcl::PointCloud<SCPointType> &_scan_down)
{

    int num_pts_scan_down = _scan_down.points.size();

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sctor_idx;
    for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_down.points[pt_idx].x;
        pt.y = _scan_down.points[pt_idx].y;
        pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).

        // xyz to ring, sector
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);

        // if range is out of roi, pass
        if (azim_range > PC_MAX_RADIUS)
            continue;

        ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
        sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), 1);

        // taking maximum z
        if (desc(ring_idx - 1, sctor_idx - 1) < pt.z) // -1 means cpp starts from 0
            desc(ring_idx - 1, sctor_idx - 1) = pt.z; // update for taking maximum value at that bin
    }

    // reset no points to zero (for cosine dist later)
    for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
        for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
            if (desc(row_idx, col_idx) == NO_POINT)
                desc(row_idx, col_idx) = 0;

    return desc;
} // SCManager::makeScancontext

MatrixXd SCManager::makeRingkeyFromScancontext(Eigen::MatrixXd &_desc)
{
    /*
     * summary: rowwise mean vector
     */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for (int row_idx = 0; row_idx < _desc.rows(); row_idx++)
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }

    return invariant_key;
} // SCManager::makeRingkeyFromScancontext

MatrixXd SCManager::makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc)
{
    /*
     * summary: columnwise mean vector
     */
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for (int col_idx = 0; col_idx < _desc.cols(); col_idx++)
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }

    return variant_key;
} // SCManager::makeSectorkeyFromScancontext

void SCManager::makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down)
{
    Eigen::MatrixXd sc = makeScancontext(_scan_down); // v1
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);

    // cout <<polarcontext_vkeys_.size() << endl;

} // SCManager::makeAndSaveScancontextAndKeys

std::pair<int, float> SCManager::detectLoopClosureID(size_t artop1, int *vals)
{
    int loop_id{-1}; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    auto curr_key = polarcontext_invkeys_mat_.back(); // current observation (query)
    auto curr_desc = polarcontexts_.back();           // current observation (query)

    /*
     * step 1: candidates from ringkey tree_
     */
    if (polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1)
    {
        std::pair<int, float> result{loop_id, 0.0};
        return result; // Early return
    }

    // tree_ reconstruction (not mandatory to make everytime)
    if (tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() - NUM_EXCLUDE_RECENT);

        polarcontext_tree_.reset();
        polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
    }
    tree_making_period_conter = tree_making_period_conter + 1;

    if (polarcontext_tree_ == nullptr)
    {
        std::pair<int, float> result{-1, 0.0};
        return result; // Early return
    }

    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    size_t NUM_CANDIDATES_FROM_TREE = artop1;

    // knn search
    std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
    knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10));

    for (size_t i = 0; i < artop1; i++)
    {
        vals[i] = candidate_indexes[i];
    }

    NUM_CANDIDATES_FROM_TREE = std::min((size_t)NUM_CANDIDATES_FROM_TREE_Init, artop1);

    /*
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    for (int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++)
    {
        MatrixXd polarcontext_candidate = polarcontexts_[candidate_indexes[candidate_iter_idx]];
        std::pair<double, int> sc_dist_result = distanceBtnScanContext(curr_desc, polarcontext_candidate);

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if (candidate_dist < min_dist)
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }

    std::pair<int, float> result{nn_idx, min_dist};

    return result;

} // SCManager::detectLoopClosureID

// } // namespace SC2

#include "provider.h"

int main(int argc, const char *const *argv)
{
    const char *seq_name = "05";
    FILE *output = stdout;

    if (argc > 1)
    {
        seq_name = argv[1];
    }

    if (argc > 2)
    {
        output = fopen(argv[2], "w");
    }

    FILE *artop1p_output = nullptr;
    if (argc > 3)
    {
        artop1p_output = fopen(argv[3], "w");
    }

    using ProviderCreatorFunc = Provider *(*)(const char *);

    ProviderCreatorFunc create_provider = create_provider_k;

    if (argc > 4)
    {
        if (strcmp(argv[4], "kitti") == 0)
        {
            create_provider = create_provider_k;
        }
        else if (strcmp(argv[4], "pcd") == 0)
        {
            create_provider = create_provider_p;
        }
    }

    Provider *provider = create_provider(seq_name);

    SCManager mgr;
    size_t full_size = provider->size();
    int artop1 = full_size / 100;
    int *vals = new int[artop1];
    for (size_t idx = 0; idx < full_size; idx++)
    {
        auto pose = provider->pose(idx);
        auto cloud = provider->cloud(idx);

        mgr.makeAndSaveScancontextAndKeys(*cloud);

        auto [loop_idx, score] = mgr.detectLoopClosureID(artop1, vals);

        if (loop_idx != -1)
        {
            fprintf(output, "%zu %d %f\n", idx, loop_idx, score);
            fprintf(artop1p_output, "%zd ", idx);
            for (int i = 0; i < artop1; i++)
            {
                fprintf(artop1p_output, "%d ", vals[i]);
            }
            fprintf(artop1p_output, "\n");
        }
    }

    fflush(output);
    if (output != stdout)
        fclose(output);

    if (artop1p_output != nullptr)
    {
        fflush(artop1p_output);
        fclose(artop1p_output);
    }

    delete provider;
    delete[] vals;
    return 0;
}