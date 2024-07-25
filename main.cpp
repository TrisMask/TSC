#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <nanoflann.hpp>
#include <array>
#include <opencv2/opencv.hpp>
#include <queue>

#include <algorithm>

struct Color3
{
	float r, g, b;

	float &operator[](int idx)
	{
		if (idx == 0)
			return r;
		if (idx == 1)
			return g;
		if (idx == 2)
			return b;
		throw std::out_of_range("Color3 index out of range");
	}
};

static Color3 rainbow_color(float value)
{
	Color3 color;
	value = std::min(value, 1.0f);
	value = std::max(value, 0.0f);

	float h = value * 5.0f + 1.0f;
	int i = floor(h);
	float f = h - i;
	if (!(i & 1))
		f = 1 - f; // if i is even
	float n = 1 - f;

	if (i <= 1)
		color[0] = n, color[1] = 0, color[2] = 1;
	else if (i == 2)
		color[0] = 0, color[1] = n, color[2] = 1;
	else if (i == 3)
		color[0] = 0, color[1] = 1, color[2] = n;
	else if (i == 4)
		color[0] = n, color[1] = 1, color[2] = 0;
	else if (i >= 5)
		color[0] = 1, color[1] = n, color[2] = 0;

	return color;
}

void save_matrix_as_image(const Eigen::MatrixXd &matrix, const std::string &filename)
{
	cv::Mat image = cv::Mat::zeros(matrix.rows(), matrix.cols(), CV_8UC3);
	double max_value = matrix(0, 0);
	double min_value = matrix(0, 0);

	for (int i = 0; i < matrix.rows(); i++)
	{
		for (int j = 0; j < matrix.cols(); j++)
		{
			max_value = std::max(max_value, matrix(i, j));
			min_value = std::min(min_value, matrix(i, j));
		}
	}

	for (int i = 0; i < matrix.rows(); i++)
	{
		for (int j = 0; j < matrix.cols(); j++)
		{
			float alpha = (matrix(i, j) - min_value) / (max_value - min_value);
			auto color = rainbow_color(1 - alpha);

			image.at<cv::Vec3b>(i, j) = cv::Vec3b(color.b * 255, color.g * 255, color.r * 255);
		}
	}

	cv::imwrite(filename, image);
}
static inline float xy2theta(const float &_x, const float &_y)
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

template <class VectorOfVectorsType, typename num_t = double, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
	typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance> self_t;
	typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
	typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;

	index_t *index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

	/// Constructor: takes a const ref to the vector of vectors object with the data points
	KDTreeVectorOfVectorsAdaptor(const size_t /* dimensionality */, const VectorOfVectorsType &mat, const int leaf_max_size = 10) : m_data(mat)
	{
		assert(mat.size() != 0 && mat[0].size() != 0);
		const size_t dims = mat[0].size();
		if (DIM > 0 && static_cast<int>(dims) != DIM)
			throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
		index = new index_t(static_cast<int>(dims), *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
		index->buildIndex();
	}

	~KDTreeVectorOfVectorsAdaptor()
	{
		delete index;
	}

	const VectorOfVectorsType &m_data;

	/** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
	 *  Note that this is a short-cut method for index->findNeighbors().
	 *  The user can also call index->... methods as desired.
	 * \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
	 */
	inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
	{
		nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
		resultSet.init(out_indices, out_distances_sq);
		index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
	}

	/** @name Interface expected by KDTreeSingleIndexAdaptor
	 * @{ */

	const self_t &derived() const
	{
		return *this;
	}
	self_t &derived()
	{
		return *this;
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const
	{
		return m_data.size();
	}

	// Returns the dim'th component of the idx'th point in the class:
	inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return m_data[idx][dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX & /*bb*/) const
	{
		return false;
	}

	/** @} */

}; // end of KDTreeVectorOfVectorsAdaptor

double distDirectSC(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2)
{
	int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
	double sum_sector_similarity = 0;
	for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++)
	{
		Eigen::VectorXd col_sc1 = _sc1.col(col_idx);
		Eigen::VectorXd col_sc2 = _sc2.col(col_idx);

		if (col_sc1.norm() < 1e-5f || col_sc2.norm() < 1e-5f)
			continue; // don't count this sector pair.

		double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

		sum_sector_similarity = sum_sector_similarity + sector_similarity;
		num_eff_cols = num_eff_cols + 1;
	}

	if (num_eff_cols == 0)
	{
		printf("num_eff_cols: %d\n", num_eff_cols);
		return 1.0;
	}

	double sc_sim = sum_sector_similarity / num_eff_cols;
	return 1.0 - sc_sim;

} // distDirectSC

static Eigen::MatrixXd circshift(const Eigen::MatrixXd &_mat, int _num_shift)
{
	// shift columns to right direction
	assert(_num_shift >= 0);

	if (_num_shift == 0)
	{
		Eigen::MatrixXd shifted_mat(_mat);
		return shifted_mat; // Early return
	}

	Eigen::MatrixXd shifted_mat = Eigen::MatrixXd::Zero(_mat.rows(), _mat.cols());
	for (int col_idx = 0; col_idx < _mat.cols(); col_idx++)
	{
		int new_location = (col_idx + _num_shift) % _mat.cols();
		shifted_mat.col(new_location) = _mat.col(col_idx);
	}

	return shifted_mat;

} // circshift

int fastAlignUsingVkey(const Eigen::MatrixXd &_vkey1, const Eigen::MatrixXd &_vkey2)
{
	int argmin_vkey_shift = 0;
	double min_veky_diff_norm = 10;
	for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++)
	{
		Eigen::MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

		Eigen::MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

		double cur_diff_norm = vkey_diff.norm();
		if (cur_diff_norm < min_veky_diff_norm)
		{
			argmin_vkey_shift = shift_idx;
			min_veky_diff_norm = cur_diff_norm;
		}
	}

	return argmin_vkey_shift;

} // fastAlignUsingVkey

Eigen::MatrixXd makeSectorkeyFromScancontext(const Eigen::MatrixXd &_desc)
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
}

std::pair<double, int> distanceBtnScanContext(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2)
{
	// 1. fast align using variant key (not in original IROS18)
	Eigen::MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
	Eigen::MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
	int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

	const int SEARCH_RADIUS = round(0.05 * _sc1.cols()); // a half of search range
	std::vector<int> shift_idx_search_space{argmin_vkey_shift};
	for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++)
	{
		shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols());
		shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols());
	}
	std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

	// 2. fast columnwise diff
	int argmin_shift = 0;
	double min_sc_dist = 1000000;
	for (int num_shift : shift_idx_search_space)
	{
		Eigen::MatrixXd sc2_shifted = circshift(_sc2, num_shift);
		double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
		if (cur_sc_dist < min_sc_dist)
		{
			argmin_shift = num_shift;
			min_sc_dist = cur_sc_dist;
		}
	}

	if (min_sc_dist > 100)
	{
		printf("min_sc_dist: %f\n", min_sc_dist);
	}

	return std::make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext]

size_t local_size = 3;
constexpr const size_t PC_NUM_RING = 20;
constexpr const size_t PC_NUM_SECTOR = 60;

constexpr const float PC_MAX_RADIUS = 80.0f;
using Desc1D = std::array<float, PC_NUM_RING>;

Desc1D makeRingkeyFromScancontext(const Eigen::MatrixXd &_desc)
{
	/*
	 * summary: rowwise mean vector
	 */
	Desc1D invariant_key;
	for (int row_idx = 0; row_idx < _desc.rows(); row_idx++)
	{
		Eigen::MatrixXd curr_row = _desc.row(row_idx);
		invariant_key[row_idx] = curr_row.mean();
	}

	return invariant_key;
} // SCManager::makeRingkeyFromScancontext

struct Mgr
{

	std::vector<pcl::PointCloud<pcl::PointXYZI>> original_clouds;
	std::vector<Eigen::Matrix4f> transforms;
	std::vector<Desc1D> descriptors;
	std::vector<Eigen::MatrixXd> desc_2ds;

	KDTreeVectorOfVectorsAdaptor<decltype(descriptors), float> *kd_tree = nullptr;

	std::queue<Eigen::MatrixXd> back_queue;

	size_t max_candidates = 10;

	FILE *artop1p_output = nullptr;

	auto check_loop(const Eigen::MatrixXd &desc)
	{
		size_t candidate_size = std::min(max_candidates, desc_2ds.size());

		Desc1D desc_1d = makeRingkeyFromScancontext(desc);
		if (kd_tree == nullptr)
		{
			return std::make_pair(size_t(-1), 0.0f);
		}
		std::vector<size_t> loop_index(candidate_size);
		std::vector<float> distance(candidate_size);

		kd_tree->query(desc_1d.data(), candidate_size, loop_index.data(), distance.data());

		if (artop1p_output != nullptr)
		{
			fprintf(artop1p_output, "%zd", original_clouds.size() - 1);
			for (size_t idx = 0; idx < candidate_size; idx++)
			{
				fprintf(artop1p_output, " %zd", loop_index[idx]);
			}
			fprintf(artop1p_output, "\n");
		}

		auto real_candidates = std::max(candidate_size, size_t(10));

		std::pair<double, int> candidate_distance = distanceBtnScanContext(desc, desc_2ds[loop_index[0]]);
		size_t candidate = 0;

		for (size_t idx = 1; idx < real_candidates; idx++)
		{
			auto [dist, shift] = distanceBtnScanContext(desc, desc_2ds[loop_index[idx]]);
			if (dist < candidate_distance.first)
			{
				candidate_distance = std::make_pair(dist, shift);
				candidate = idx;
			}
		}

		return std::make_pair(loop_index[candidate], (float)candidate_distance.first);
	}

	auto candidates(size_t count)
	{
		max_candidates = count;
	}

	auto push_object(const pcl::PointCloud<pcl::PointXYZI> &cloud, Eigen::Matrix4f tr)
	{
		original_clouds.push_back(std::move(cloud));
		transforms.push_back(std::move(tr));

		Eigen::MatrixXd desc_2d = generate_descriptor();

		auto loop_index = check_loop(desc_2d);

		back_queue.push(std::move(desc_2d));

		if (back_queue.size() >= 100)
		{
			while (back_queue.size() >= 50)
			{
				desc_2ds.push_back(back_queue.front());
				descriptors.push_back(makeRingkeyFromScancontext(back_queue.front()));
				back_queue.pop();
			}
			build_index();
		}

		return loop_index;
	}

	Eigen::MatrixXd generate_descriptor()
	{
		constexpr float NO_POINT = -1000.0f;
		Eigen::MatrixXd desc = Eigen::MatrixXd::Constant(PC_NUM_RING, PC_NUM_SECTOR, NO_POINT);

		size_t start_index = original_clouds.size() > local_size ? original_clouds.size() - local_size : 0;

		for (size_t idx = start_index; idx < original_clouds.size(); idx++)
		{
			auto &_scan_down_origin = original_clouds[idx];
			pcl::PointCloud<pcl::PointXYZI> _scan_down;

			pcl::transformPointCloud(_scan_down_origin, _scan_down, transforms.back().inverse() * transforms[idx]);
			for (int pt_idx = 0; pt_idx < _scan_down.size(); pt_idx++)
			{
				pcl::PointXYZ pt;
				pt.x = _scan_down.points[pt_idx].x;
				pt.y = _scan_down.points[pt_idx].y;
				pt.z = _scan_down.points[pt_idx].z + 2.0f; // naive adding is ok (all points should be > 0).

				// xyz to ring, sector
				float azim_range = sqrtf(pt.x * pt.x + pt.y * pt.y);
				float azim_angle = xy2theta(pt.x, pt.y);

				// if range is out of roi, pass
				if (azim_range > PC_MAX_RADIUS)
					continue;

				auto ring_idx = std::max(std::min(PC_NUM_RING, size_t(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), size_t(1));
				auto sctor_idx = std::max(std::min(PC_NUM_SECTOR, size_t(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), size_t(1));

				// taking maximum z
				if (desc(ring_idx - 1, sctor_idx - 1) < pt.z) // -1 means cpp starts from 0
					desc(ring_idx - 1, sctor_idx - 1) = pt.z; // update for taking maximum value at that bin
			}
		}

		for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
			for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
				if (desc(row_idx, col_idx) == NO_POINT)
					desc(row_idx, col_idx) = 0;

		return desc;
	}

	void build_index()
	{
		if (kd_tree != nullptr)
		{
			kd_tree->index->buildIndex();
		}
		else
		{
			kd_tree = new KDTreeVectorOfVectorsAdaptor<decltype(descriptors), float>(PC_NUM_RING, descriptors);
		}
	}
};

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

	if (argc > 5)
	{
		int local_size_i = atoi(argv[5]);
		if (local_size_i > 0)
		{
			local_size = local_size_i;
		}
	}

	Provider *provider = create_provider(seq_name);

	Mgr mgr;
	mgr.artop1p_output = artop1p_output;
	size_t full_size = provider->size();
	mgr.candidates(full_size / 100);

	for (size_t idx = 0; idx < full_size; idx++)
	{
		auto pose = provider->pose(idx);
		auto cloud = provider->cloud(idx);
		auto [loop_idx, score] = mgr.push_object(*cloud, pose);

		if (loop_idx != size_t(-1))
			fprintf(output, "%zd %zd %f\n", idx, loop_idx, score);
	}

	fflush(output);
	if (output != stdout)
		fclose(output);
	return 0;
}