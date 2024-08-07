/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @文件 ImuFactorsExample
 * @简述 使用GTSAM的ImuFactor和ImuCombinedFactor进行导航代码的测试示例。
 * @作者 Garrett (ghemann@gmail.com), Luca Carlone
 */

/**
 * 使用imuFactors（imuFactor和combinedImuFactor）与GPS结合的示例
 *  - 默认使用imuFactor。可以通过添加`-c`标志来测试combinedImuFactor（参见下方的示例命令）。
 *  - 我们从CSV文件读取IMU和GPS数据，格式如下：
 *  一行以"i"开头，表示初始位置，格式为N, E, D, qx, qY, qZ, qW, velN, velE, velD
 *  一行以"0"开头，表示IMU测量数据（坐标系 - 前进，右，下）
 *  linAccX, linAccY, linAccZ, angVelX, angVelY, angVelX
 *  一行以"1"开头，表示GPS修正，格式为N, E, D, qX, qY, qZ, qW
 *  注意，对于GPS修正，我们仅使用位置，不使用旋转。文件中提供旋转数据用于与真实数据对比。
 *
 *  查看用法：./ImuFactorsExample --help
 */

#include <boost/program_options.hpp>

// GTSAM related includes.
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <cstring>
#include <fstream>
#include <iostream>

using namespace gtsam;
using namespace std;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

namespace po = boost::program_options;

/**
 * 解析命令行选项并返回一个变量映射。
 *
 * @param argc 命令行参数的数量。
 * @param argv 命令行参数的字符串数组。
 * @return 返回解析后的变量映射(po::variables_map)。
 *
 * 函数设置命令行选项，并解析提供的参数。支持的选项包括：
 * - help,h: 显示帮助信息。
 * - data_csv_path: IMU数据CSV文件的路径，默认为"imuAndGPSdata.csv"。
 * - output_filename: 结果文件的路径，默认为"imuFactorExampleResults.csv"。
 * - use_isam: 是否使用ISAM作为优化器，默认关闭。
 *
 * 如果用户请求帮助信息，函数将显示可用的选项并退出程序。
 */
po::variables_map parseOptions(int argc, char *argv[])
{
  po::options_description desc;                        // 创建选项描述对象
  desc.add_options()("help,h", "produce help message") // 添加一个选项“help”，短标识为“h”，当使用此选项时，程序将输出帮助信息。
      ("data_csv_path", po::value<string>()->default_value("imuAndGPSdata.csv"),
       "path to the CSV file with the IMU data") // 添加一个选项“data_csv_path”，此选项需要一个字符串值，指定IMU数据的CSV文件路径，默认值为“imuAndGPSdata.csv”。
      ("output_filename",
       po::value<string>()->default_value("imuFactorExampleResults.csv"),
       "path to the result file to use") // 添加一个选项“output_filename”，此选项需要一个字符串值，指定输出结果文件的路径，默认值为“imuFactorExampleResults.csv”。
      ("use_isam", po::bool_switch(),
       "use ISAM as the optimizer"); // 添加一个选项“use_isam”，此选项是一个布尔开关，用来决定是否使用ISAM作为优化器，默认为关闭状态。

  po::variables_map vm;                                    // 存储解析后的选项数据
  po::store(po::parse_command_line(argc, argv, desc), vm); // 解析命令行参数

  if (vm.count("help")) // 如果请求了帮助
  {
    cout << desc << "\n"; // 输出帮助信息
    exit(1);
  }

  return vm; // 返回解析的选项数据
}

// 设置IMU预积分参数，定义加速度计和陀螺仪的噪声模型，以及加速度计和陀螺仪的偏差随机游走噪声
boost::shared_ptr<PreintegratedCombinedMeasurements::Params> imuParams()
{
  // We use the sensor specs to build the noise model for the IMU factor.
  double accel_noise_sigma = 0.0003924;
  double gyro_noise_sigma = 0.000205689024915;
  double accel_bias_rw_sigma = 0.004905;
  double gyro_bias_rw_sigma = 0.000001454441043;
  Matrix33 measured_acc_cov = I_3x3 * pow(accel_noise_sigma, 2);
  Matrix33 measured_omega_cov = I_3x3 * pow(gyro_noise_sigma, 2);
  Matrix33 integration_error_cov =
      I_3x3 * 1e-8; // error committed in integrating position from velocities
  Matrix33 bias_acc_cov = I_3x3 * pow(accel_bias_rw_sigma, 2);
  Matrix33 bias_omega_cov = I_3x3 * pow(gyro_bias_rw_sigma, 2);
  Matrix66 bias_acc_omega_init =
      I_6x6 * 1e-5; // error in the bias used for preintegration

  auto p = PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);
  // PreintegrationBase params:
  p->accelerometerCovariance =
      measured_acc_cov; // acc white noise in continuous
  p->integrationCovariance =
      integration_error_cov; // integration uncertainty continuous
  // should be using 2nd order integration
  // PreintegratedRotation params:
  p->gyroscopeCovariance =
      measured_omega_cov; // gyro white noise in continuous
  // PreintegrationCombinedMeasurements params:
  p->biasAccCovariance = bias_acc_cov;     // acc bias in continuous
  p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
  p->biasAccOmegaInt = bias_acc_omega_init;

  return p;
}

int main(int argc, char *argv[])
{

  // 解析命令行参数，设置数据和输出文件路径，以及是否使用ISAM优化器
  po::variables_map var_map = parseOptions(argc, argv);

  string data_filename = var_map["data_csv_path"].as<string>();     // IMU和GPS数据文件路径
  string output_filename = var_map["output_filename"].as<string>(); // 结果输出文件路径
  bool use_isam = var_map["use_isam"].as<bool>();                   // 标记是否使用ISAM2作为优化器

  // 创建ISAM2或Levenberg Marquardt优化器的实例
  ISAM2 *isam2 = 0;
  if (use_isam)
  {
    printf("Using ISAM2\n");
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam2 = new ISAM2(parameters);
  }
  else
  {
    printf("Using Levenberg Marquardt Optimizer\n");
  }

  // 打开输出文件，用于记录优化的结果
  FILE *fp_out = fopen(output_filename.c_str(), "w+");
  fprintf(fp_out,
          "#time(s),x(m),y(m),z(m),qx,qy,qz,qw,gt_x(m),gt_y(m),gt_z(m),gt_qx,"
          "gt_qy,gt_qz,gt_qw\n");

  // 开始从CSV文件读取数据，并初始化IMU或添加GPS数据
  ifstream file(data_filename.c_str());
  string value;

  // Format is (N,E,D,qX,qY,qZ,qW,velN,velE,velD)
  // 读取并解析初始状态
  Vector10 initial_state;
  getline(file, value, ','); // 跳过行首标记'i'
  for (int i = 0; i < 9; i++)
  {
    getline(file, value, ',');
    initial_state(i) = stof(value.c_str());
  }
  getline(file, value, '\n');
  initial_state(9) = stof(value.c_str());
  cout << "initial state:\n"
       << initial_state.transpose() << "\n\n";

  // 设置初始的姿态和位置
  Rot3 prior_rotation = Rot3::Quaternion(initial_state(6), initial_state(3),
                                         initial_state(4), initial_state(5));
  Point3 prior_point(initial_state.head<3>());
  Pose3 prior_pose(prior_rotation, prior_point);
  Vector3 prior_velocity(initial_state.tail<3>());
  imuBias::ConstantBias prior_imu_bias; // 假设初始偏差为零

  // 创建一个 Values 对象，用来存储优化问题的初始估计值。
  Values initial_values;
  int correction_count = 0;   // 用作因子图中的键索引，表示当前处理的是第几个状态估计。
  // X(correction_count) 生成一个与 correction_count 相关联的符号键，用于表示机器人或传感器的位姿（位置和方向）。
  // 在 GTSAM 中，符号键通常由一个字母（如 X 表示位姿，V 表示速度，B 表示偏置）和一个数字组成，数字用来区分不同的时间步或实例。
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_velocity);
  initial_values.insert(B(correction_count), prior_imu_bias);

  // 添加先验噪声模型并构建因子图
  auto pose_noise_model = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5)
          .finished());                                             // rad,rad,rad,m, m, m
  auto velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1); // m/s
  auto bias_noise_model = noiseModel::Isotropic::Sigma(6, 1e-3);

  // 添加所有的先验因子和噪声模型
  NonlinearFactorGraph *graph = new NonlinearFactorGraph();
  graph->addPrior(X(correction_count), prior_pose, pose_noise_model);
  graph->addPrior(V(correction_count), prior_velocity, velocity_noise_model);
  graph->addPrior(B(correction_count), prior_imu_bias, bias_noise_model);

  // 设置IMU预积分参数
  auto p = imuParams();

  std::shared_ptr<PreintegrationType> preintegrated =
      std::make_shared<PreintegratedImuMeasurements>(p, prior_imu_bias);

  assert(preintegrated);

  // 从文件中读取数据，并应用IMU或GPS的更新
  NavState prev_state(prior_pose, prior_velocity);
  NavState prop_state = prev_state;
  imuBias::ConstantBias prev_bias = prior_imu_bias;

  // Keep track of total error over the entire run as simple performance metric.
  double current_position_error = 0.0, current_orientation_error = 0.0;

  double output_time = 0.0;
  // 使用固定的时间间隔
  double dt = 0.005; // The real system has noise, but here, results are nearly
                     // exactly the same, so keeping this for simplicity.

  // All priors have been set up, now iterate through the data file.
  while (file.good())
  {
    // 解析一行数据，区分IMU或GPS数据
    getline(file, value, ',');
    int type = stoi(value.c_str());

    if (type == 0)
    {
      // IMU数据
      Vector6 imu;
      for (int i = 0; i < 5; ++i)
      {
        getline(file, value, ',');
        imu(i) = stof(value.c_str());
      }
      getline(file, value, '\n');
      imu(5) = stof(value.c_str());

      // 添加IMU预积分
      preintegrated->integrateMeasurement(imu.head<3>(), imu.tail<3>(), dt);
    }
    else if (type == 1)
    { 
      // GPS数据
      Vector7 gps;
      for (int i = 0; i < 6; ++i)
      {
        getline(file, value, ',');
        gps(i) = stof(value.c_str());
      }
      getline(file, value, '\n');
      gps(6) = stof(value.c_str());

      correction_count++;

      // 添加IMU和GPS因子，并进行优化
      auto preint_imu =
          dynamic_cast<const PreintegratedImuMeasurements &>(*preintegrated);
      ImuFactor imu_factor(X(correction_count - 1), V(correction_count - 1),
                           X(correction_count), V(correction_count),
                           B(correction_count - 1), preint_imu);
      graph->add(imu_factor);
      imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
      graph->add(BetweenFactor<imuBias::ConstantBias>(
          B(correction_count - 1), B(correction_count), zero_bias,
          bias_noise_model));

      auto correction_noise = noiseModel::Isotropic::Sigma(3, 1.0);
      GPSFactor gps_factor(X(correction_count),
                           Point3(gps(0),  // N,
                                  gps(1),  // E,
                                  gps(2)), // D,
                           correction_noise);
      graph->add(gps_factor);

      // 执行优化并比较结果
      prop_state = preintegrated->predict(prev_state, prev_bias);
      initial_values.insert(X(correction_count), prop_state.pose());
      initial_values.insert(V(correction_count), prop_state.v());
      initial_values.insert(B(correction_count), prev_bias);

      Values result;

      if (use_isam)
      {
        isam2->update(*graph, initial_values);
        result = isam2->calculateEstimate();

        // 重置因子图和初始值
        graph->resize(0);
        initial_values.clear();
      }
      else
      {
        LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
        result = optimizer.optimize();
      }

     // 重新设置预积分器的bias
      prev_state = NavState(result.at<Pose3>(X(correction_count)),
                            result.at<Vector3>(V(correction_count)));
      prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));

      // Reset the preintegration object.
      preintegrated->resetIntegrationAndSetBias(prev_bias);

     // 记录位置和方向误差
      Vector3 gtsam_position = prev_state.pose().translation();
      Vector3 position_error = gtsam_position - gps.head<3>();
      current_position_error = position_error.norm();

      Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
      Quaternion gps_quat(gps(6), gps(3), gps(4), gps(5));
      Quaternion quat_error = gtsam_quat * gps_quat.inverse();
      quat_error.normalize();
      Vector3 euler_angle_error(quat_error.x() * 2, quat_error.y() * 2,
                                quat_error.z() * 2);
      current_orientation_error = euler_angle_error.norm();

   // 输出统计信息
      cout << "Position error:" << current_position_error << "\t "
           << "Angular error:" << current_orientation_error << "\n";

      fprintf(fp_out, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
              output_time, gtsam_position(0), gtsam_position(1),
              gtsam_position(2), gtsam_quat.x(), gtsam_quat.y(), gtsam_quat.z(),
              gtsam_quat.w(), gps(0), gps(1), gps(2), gps_quat.x(),
              gps_quat.y(), gps_quat.z(), gps_quat.w());

      output_time += 1.0;
    }
    else
    {
      cerr << "ERROR parsing file\n";
      return 1;
    }
  }
  fclose(fp_out);
  cout << "Complete, results written to " << output_filename << "\n\n";

  return 0;
}
