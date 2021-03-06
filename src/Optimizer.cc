﻿/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    //typedef BlockSolver< BlockSolverTraits<6, 3> > BlockSolver_6_3;
    //这表明误差变量为6维，误差项为3维
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    //是否开启强制停止开关
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    //遍历提供的所有关键帧，向g2o中添加顶点误差变量，为keyframe里的相机位姿
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        //节点类型为g2o::VertexSE3Expmap
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        //设置位姿顶点误差变量的初始值
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));    //以关键帧的大概位姿作为初始值
        //设置顶点ID，为关键帧ID
        vSE3->setId(pKF->mnId);
        //如果是第0帧，那么就不优化这个顶点误差变量(固定第0帧，避免零空间漂移)
        vSE3->setFixed(pKF->mnId==0);
        //将配置好的顶点添加到optimizer
        optimizer.addVertex(vSE3);
        //更新maxKFid(最大id号？)
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    //核函数相关参数
    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    // 所有地图点也作为顶点，作为待优化变量
    // 遍历vpMP提供的所有mappoint，向g2o添加顶点误差变量
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        //顶点类型为g2o::VertexSBAPointXYZ
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        //设定顶点的初始值(3D点的世界坐标)
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        //注意这里和位姿顶点的ID向匹配
        const int id = pMP->mnId+maxKFid+1; //顶点ID在位姿顶点之后
        vPoint->setId(id);
        //设置该点在解方程时进行schur消元，就是是否利用稀疏化加速
        vPoint->setMarginalized(true);
        //将配置好的顶点添加到optimizer
        optimizer.addVertex(vPoint);

        //获取观测到这个路标点的所有关键帧以及对应的特征点
        //observations: 是一组映射<关键帧，对应的特征点idx>
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        //遍历此mappoint能被看到的所有keyframe，向优化器添加误差边
        //遍历所有观测到这一个路标点的关键帧，增加重投影误差作为约束
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            //取关键帧
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;
            //取特征点(u,v)
            //mit->second：特征点idx
            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            //开始添加边
            // 单目或RGBD相机
            if(pKF->mvuRight[mit->second]<0)    //单目的情况下，mvuRight[]为负值
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                //边类型为g2o::EdgeSE3ProjectXYZ
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                //添加和这条边相连的mappoint顶点，0表示是mappoint类型，
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                //添加和这条边相连的位姿顶点，也就是观测到这个路标点的关键帧对应的位姿顶点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                //观测为这个关键帧下对应的特征点
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                //根据mappoint所在高斯金字塔尺度设置信息矩阵
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                //如果需要开启核函数
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                //向边添加内参
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                //添加边
                optimizer.addEdge(e);
            }
            //双目
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        //如果当前这个mappoint没有被任何关键帧观测到
        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint); //移除，不参与优化
            vbNotIncludedMP[i]=true;    //对应标志位
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    //  Recover optimized data
    //  从优化结果获取数据
    //  Keyframes
    //  遍历所有关键帧，根据优化结果更新关键帧的位姿
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        //这里是确保数据类型正确
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        //取出优化变量vSE3的结果
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            //取顶点位姿，更新对应关键帧位姿
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            //将全局BA优化的结果存到mTcwGBA，并且设置
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            //这个标志是LoopCLosing.cc 里面 执行完全局BA之后，对所有关键帧进行位姿设置的时候用的
            //如果这个值没有被设置，表示是Local Mapping新插进来的关键帧，没有经过这次的BA优化
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    //遍历取出优化变量结果，更新mappoint
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        //取出顶点优化变量g2o::VertexSBAPointXYZ结果
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            //更新mappoint位置
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            //
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            //这个标志是LoopCLosing.cc 里面 执行完全局BA之后，对所有关键帧进行位姿设置的时候用的
            //如果这个值没有被设置，表示是Local Mapping新插进来的关键帧，没有经过这次的BA优化
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

//前端BA
int Optimizer::PoseOptimization(Frame *pFrame)
{
    //这里请参考Optimizer::BundleAdjustment的注释
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;  //记录边的数量,也就是关键点的数量

    // Set Frame vertex
    //将pFrame的位姿添加为顶点作为优化变量
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw)); //cv::Mat 到 g2o::SE3Quat类型转换
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;//关键点数量,也是特征点所构建的边的数量

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono; //单目的边
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    //鲁棒性核函数的参数
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        //遍历pFrame帧的所有特征点，添加g2o边
        for(int i=0; i<N; i++)
        {
            //获取当前特征点所对应的路标点(世界坐标系3D点\地图点)
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            //如果此特征点有对应的mappoint
            if(pMP)
            {
                // Monocular observation
                //单目
                if(pFrame->mvuRight[i]<0)   //负值表示单目观测
                {
                    //记录添加了多少条边
                    nInitialCorrespondences++;
                    //先将这个特征点设置为不是Outlier
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;  //观测,就是去畸变后的特征点坐标
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    //注意这里的边与GlobalBundleAdjustemnt()边不一样
                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                    //设置顶点,这里就一个顶点,就是当前帧的相机pose
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs); //设置观测,也就是当前索引为i的去畸变后的关键点的坐标
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];  //INFO
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;    //鲁棒性核函数Huber
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    //设置相机内参
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;

                    //获取关键点在世界坐标系的坐标
                    //也就是路标点的世界坐标
                    //重投影的时候需要将这个点经过相机位姿投影到图像上,与观测做残差
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation  双目的观测
                {
                    //记录添加了多少条边
                    nInitialCorrespondences++;
                    //先将这个特征点设置为不是Outlier
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    //双目观测的边
                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }

        }
    }

    //如果只添加了3条边
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值(查表)
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};      //(0.05概率下偏差2个单位)
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};   //(0.05概率下偏差3个单位)
    const int its[4]={10,10,10,10};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        //设置初始值,这个初始值实际上是上个参考关键帧的位姿
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        //启动优化
        optimizer.optimize(its[it]); //每次优化迭代10次

        nBad=0;
        //遍历单目模式的每条边
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)   //每次优化完都进入到这里遍历
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i]; //取一条边

            const size_t idx = vnIndexEdgeMono[i];  //取边的id

            if(pFrame->mvbOutlier[idx]) //第一次的时候,这个数组全为false
            {
                e->computeError(); //计算这条边的残差
            }

            const float chi2 = e->chi2(); //平方

            if(chi2>chi2Mono[it])  //判断残差是否大与阈值
            {
                pFrame->mvbOutlier[idx]=true; //则设置这条边为outlier
                e->setLevel(1); //设置这条边的等级?
                nBad++;         //坏的边数+1
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0); //如果是第二次优化了,则不用核函数?
        }

        //遍历双目模式的每条边
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            //没懂
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            //判断残差是否大与阈值
            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        //如果边少于10,则不优化
        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    // 获取优化后的结果,返回有效的边(关键点)数目
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov); //位姿重新转回cv::Mat类型
    pFrame->SetPose(pose);  //设置该帧的pose

    return nInitialCorrespondences-nBad;//返回好的边(关键点)数
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // 将当前关键帧push
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    //将当前关键帧pKF的共视图中与pKF连接的关键帧放入lLocalKeyFrames
    //pKF->GetVectorCovisibleKeyFrames()：返回与此关键帧具有连接关系的关键帧，其顺序按照共视的mappoint数量递减排序
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // 将被lLocalKeyFrames所有关键帧看到的mappoint放入lLocalMapPoints中
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // lFixedCameras储存着能看到lLocalMapPoints，但是又不在lLocalKeyFrames里的关键帧
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        //取能观测到这个mappoint的keyframe(不一定在lLocalKeyFrames里面)
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // 将lLocalKeyFrames关键帧的位姿设置为g2o图的顶点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    // 将lFixedCameras里的关键帧的位姿设置为g2o图的顶点(这些顶点是固定的顶点)
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    //Huber核函数参数
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    //将lLocalMapPoints里的mappoint空间位置作为g2o图的顶点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);  //边缘化，消元求解
        optimizer.addVertex(vPoint);

        //取观测到mappoint的KeyFrame
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        //遍历当前mappoint的每个观测
        //将这些观测形成g2o图的一条边，以重投影误差作为误差项
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)       //单目
                {
                    //取这个mappoint在pKFi的对应特征点作为观测
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    //由于观测到mappoint的每个keyframe都被插入作为顶点了，所以这里直接连接顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    //双目
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    // 检查pbStopFlag指针有没有值
    if(pbStopFlag)
        if(*pbStopFlag) //检查pbStopFlag标志，如果要求停止，则直接返回，不优化
            return;

    //开始优化，先迭代5次
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    // 再次检查pbStopFlag指针有没有值
    if(pbStopFlag)
        if(*pbStopFlag) //检查pbStopFlag标志，如果要求停止，则直接返回，终止优化
            bDoMore = false;

    // 如果还是没有请求停止的标志，则继续优化
    if(bDoMore)
    {

        // Check inlier observations
        // 检查inlier
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            // 取单目的边以及对应的mappoint
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;
            //检查这条边的误差，卡方校验
            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                //这条边是outlier ?
                e->setLevel(1);
            }
            //取消使用核函数
            e->setRobustKernel(0);
        }
        // 双目
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        // 再次优化，已经剔除了被标记为outlier的边 (通过设置边的level)
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    // 根据优化结果，选择需要提出的关键帧和mappoint
    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // 单目
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        // 取单目的边
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;
        // 卡方校验
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            //超过阈值，加入剔除队列
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    //双目
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    // 获取线程锁
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 剔除
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // 利用优化完的结果进行更新
    // Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // pLoopKF：闭环关键帧
    // pCurKF：当前关键帧
    // NonCorrectedSim3：当前关键帧以及共视的关键帧的原位姿
    // CorrectedSim3：当前关键帧以及共视的关键帧的经过sim3修正的位姿
    // LoopConnections: 新的共视连接关系

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // typedef BlockSolver< BlockSolverTraits<7, 3> > BlockSolver_7_3;
    //这表明误差变量为7维，误差项为3维
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    // 取地图所有关键帧和mappoint
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();   //取地图最新关键帧id

    //储存ba优化前的原sim3位姿
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    //
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    //储存所有顶点
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    //将map中的所有关键帧添加为g2o的顶点
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        //寻找关键帧pKF的sim3修正位姿
        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        //表示CorrectedSim3.find(pKF)寻找成功
        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            // 使用原来的sim3位姿作为初始值
            VSim3->setEstimate(it->second);
        }
        else
        {
            //如果当前的关键帧没有sim3位姿
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            //则取其se3位姿，然后把尺度设置为1
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        //如果遍历到的关键帧是闭环关键帧，则固定其顶点sim3位姿
        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        //顶点id设置为关键帧id
        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    //在g2o中已经形成误差边的两个顶点，firstid数较小的顶点
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    // LoopConnections: 新的共视连接关系
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        //取一个关键帧
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        //取关键帧pKF的共视关键帧集合spConnections
        const set<KeyFrame*> &spConnections = mit->second;
        //vScw： 储存ba优化前的原sim3位姿
        const g2o::Sim3 Siw = vScw[nIDi];
        // 取逆，即为关键帧pKF的相机坐标系到世界坐标系的sim3变换
        const g2o::Sim3 Swi = Siw.inverse();

        //遍历关键帧pKF的共视关键帧集合spConnections
        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            //取集合中的一个关键帧id_j
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;
            //取该id对应的sim3位姿
            const g2o::Sim3 Sjw = vScw[nIDj];
            //关键帧i与j之间的相对sim3变换，得到从关键帧i到关键帧j的sim3变换
            const g2o::Sim3 Sji = Sjw * Swi;

            // g2o sim3类型的边
            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            // 添加关键帧i和关键j对应的顶点作为连接顶点
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            // 观测量即为上面根据两帧sim3位姿得到的相对变换
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    // vpKFs: 地图上所有关键帧
    // 遍历vpKFs，将vpKFs和其在spanningtree中的父节点在g2o图中连接起来形成一条误差边；
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        //取关键帧pKF
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        //检查关键帧pKF是否在之前得到了sim3位姿
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();  //没有，则取原来的位姿Tcw的逆(没有经过sim3修正)，尺度为1
        else
            Swi = vScw[nIDi].inverse();     //否则，取sim3位姿

        KeyFrame* pParentKF = pKF->GetParent(); //取pKF父节点，即共视程度最高的关键帧

        // Spanning tree edge
        //将vpKFs和其在spanningtree中的父节点在g2o图中连接起来形成一条误差边；
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;
            //取父节点关键帧的sim3位姿
            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;  //没有，则取原来的位姿Tcw的逆(没有经过sim3修正)，尺度为1
            else
                Sjw = vScw[nIDj];   //否则，取sim3位姿

            //得到关键帧pKF与父节点关键帧的sim3相对变换，作为观测值
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 取地图的关键帧已经构造的回环边mspLoopEdges，这个在OptimizeEssentialGraph才被设置
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        // vpKFs: 地图上所有关键帧
        // 将vpKFs和其形成闭环的帧在g2o图中连接起来形成一条误差边
        // 说白了就是把已经形成回环的记录下来了，这里直接取这些已经是回环的边
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            //取与关键帧pKF形成回环的一个关键帧pLKF
            KeyFrame* pLKF = *sit;
            //检查先后次序
            if(pLKF->mnId<pKF->mnId)
            {
                //构造观测
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];
                g2o::Sim3 Sli = Slw * Swi;

                //生成g2o边
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        //pKF与在Covisibility graph中与pKF连接，且共视点超过minFeat的关键帧，形成一条误差边（如果之前没有添加过的话）
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            //避免和前面的边添加重复
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    //为避免重复添加，先查找
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;
                    //构造观测
                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];
                    g2o::Sim3 Sni = Snw * Swi;

                    //构造g2o边
                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 更新优化后的闭环检测位姿
    for(size_t i=0;i<vpKFs.size();i++)
    {
        //取地图的关键帧
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        //取优化之后的顶点
        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        //得到优化之后的关键帧sim3位姿
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        //vCorrectedSwc： 储存逆的形式，即 从相机坐标系到世界坐标系的sim3变换
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

        //取优化结果更新关键帧位姿
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();
        //这是为了保持尺度，与mappoint保持一致，mappoint的修正经历两次sim3投影，一次正一次反
        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 修正mappoint位姿，先使用没有优化的位姿投影到参考关键帧，再使用该关键帧优化之后的sim3位姿投影回来
    // 经历两次投影，一次正，一次反，所以尺度不会发生很大变化
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        // 检查该mappoint是否在CorrLoop主函数中修正过
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            //是，则直接取修正的关键帧id
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            //否则，取生成这个mappoint的关键帧的id
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        //ba优化前的原sim3位姿
        g2o::Sim3 Srw = vScw[nIDr];
        //ba优化后的sim3位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];
        //取当前mappoint坐标
        cv::Mat P3Dw = pMP->GetWorldPos();
        //转化为EIgen
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        //correctedSwr=[s2,R2,t2]
        //1.先使用优化前的原sim3位姿将mappoint进行变换 p_s1=s1R1* p_w + t1
        //2.再使用优化后的sim3位姿，将mappoint反变换到世界坐标系 p_w_new=s2R2* p_s1+ t2
        //其中，s2=1/s_opt R2=R_opt^T  t2=-s2*R2*t_opt
        //所以，如果s1=s_opt R1=R_opt t1=t_opt (即优化值等于初始值)，那么 p_w_new = p_w
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();//更新此mappoint参考帧光心到mappoint平均观测方向以及观测距离范围
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    //  typedef BlockSolver< BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> > BlockSolverX;
    //这表明误差变量和误差项的维度是动态的
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    // 内参(mk应该是经过畸变矫正的内参)
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    // 位姿： 从世界坐标系到相机坐标系的变换
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    // 添加sim3位姿顶点误差变量
    // g2oS12: 表示候选帧pKF2到当前帧pKF1的Sim3变换
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    //设置顶点的初始值
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    //将内参导入顶点
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    //获得当前帧pKF1的所有mappoint
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    //将两帧的mappoint转换到各自相机坐标系下，作为g2o的固定的顶点
    //添加边
    //一次可以构成两条边:
    // 1. 将闭环关键帧pKF2的mappoint投影到相机1的图像帧上与当前帧pKF1特征点的误差构成一条边
    // 2. 将当前帧pKF1的mappoint投影到相机2的图像帧上与闭环关键帧pKF2特征点的误差构成另一条边
    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];   //取当前帧pKF1的mappoint
        MapPoint* pMP2 = vpMatches1[i];     //取闭环关键帧pKF2的mappoint

        // 两个顶点两个顶点的加进去
        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);
        // 检查pMP2是否在闭环关键帧pKF2的观测中
        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                //添加当前帧pKF1的mappoint在相机1坐标系的坐标作为固定的顶点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                //当前帧pKF1的mappoint转换到pKF1相机1坐标系
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                //顶点设置
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);    //固定
                optimizer.addVertex(vPoint1);

                //添加闭环关键帧pKF2的mappoint在相机2坐标系的坐标作为固定的顶点
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                //闭环关键帧pKF2的mappoint转换到相机2坐标系
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                //顶点设置
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);    //固定
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        // 添加误差项边
        // 观测值为当前帧对应的特征点(u,v)
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        // 将e12边和vertex(id2)绑定
        // 顶点(id2): 闭环关键帧pKF2的mappoint在相机2坐标系的坐标作为顶点
        // g2o里面定义求偏差：将pKF2的mappoint在相机2坐标系的坐标使用sim3
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // 设定观测值
        // 观测值为当前帧对应的特征点(u,v)
        e12->setMeasurement(obs1);
        // info
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);
        // 鲁棒核函数
        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        // 将边添加都求解器
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        //注意这个的边类型和上面不一样
        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        //保存这些边
        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    // 把不是inliner的边剔除出去
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        //遍历储存下来的边
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        //只要对应索引i的两条边任意一条边的误差超过校验值，都认为是outlier
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    //只要有坏的边，则接下来再优化10次
    //否则5次
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    //nCorrespondences: 包括坏边的全部匹配，如果outlier边占的太多了，直接返回
    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    // 剔除边后再次优化
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    //看哪些匹配是inliner
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        //剔除outlier的边
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    // 取优化后的结果
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
