/**
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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        // 如果有新的keyframe插入到闭环检测序列（在localmapping::run()结尾处插入）
        if(CheckNewKeyFrames())
        {
            //如果队列中有新的关键帧插入


            // Detect loop candidates and check covisibility consistency
            // 检测是否有闭环(选出候选关键帧，检查共视一致性)
            if(DetectLoop())
            {
                // Compute similarity transformation [sR|t]
                // In the stereo/RGBD case s=1 对于双目或者RGBD，尺度因子=1

                //计算候选关键帧的与当前帧的sim3并且返回是否形成闭环的判断
                //并在候选帧中找出闭环帧
                //并计算出当前帧和闭环帧的sim3
                if(ComputeSim3())
                {
                    // 找到匹配的，可以形成闭环的候选关键帧(闭环帧)
                    // Perform loop fusion and pose graph optimization
                    // 纠正，融合，优化
                    CorrectLoop();
                }
            }
        }
        //检查是否有请求重置，
        ResetIfRequested();
        //检查是否请求完成停止
        if(CheckFinish())
            break;

        usleep(5000);
    }
    //设置完成停止标志
    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}


bool LoopClosing::DetectLoop()
{
    //先将要处理的闭环检测队列的关键帧弹出来一个
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        // 避免这个关键帧被这个线程处理的时候被擦除掉，设置标志位
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 步骤1：如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    // 步骤2：遍历所有共视关键帧，计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分minScore
    
    //返回Covisibility graph中与此节点连接的节点（即关键帧），总的来说这一步是为了计算阈值 minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 步骤3：在所有关键帧中找出闭环备选帧
    // 在最低相似度 minScore的要求下，获得闭环检测的候选帧集合
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    //vpCandidateKFs中的每个闭环检测的候选帧都会，通过共视关键帧，扩展为一个spCandidateGroup
    //对于这vpCandidateKFs.size()个spCandidateGroup进行连续性（consistency）判断
    
    // 步骤4：在候选帧中检测具有连续性的候选帧
    // 1、每个候选帧将与自己相连的关键帧构成一个“子候选组spCandidateGroup”，vpCandidateKFs-->spCandidateGroup
    // 2、检测“子候选组”中每一个关键帧是否存在于“连续组”，如果存在nCurrentConsistency++，则将该“子候选组”放入“当前连续组vCurrentConsistentGroups”
    // 3、如果nCurrentConsistency大于等于3，那么该”子候选组“代表的候选帧过关，进入mvpEnoughConsistentCandidates
    
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    //遍历vpCandidateKFs，将其中每个关键帧都通过寻找在covisibility graph与自己连接的关键帧，扩展为一个spCandidateGroup
    //也就是遍历每一个spCandidateGroup
    //FOR1
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        // 将自己以及与自己相连的关键帧构成一个“子候选组”
        //这个条件是否太宽松?pCandidateKF->GetVectorCovisibleKeyFrames()是否更好一点？
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        //
        bool bConsistentForSomeGroup = false;

        //遍历mvConsistentGroups，判断spCandidateGroup与mvConsistentGroups[iG]是否连续
        //FOR2
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            //当前的spCandidateGroup之后要不要插入vCurrentConsistentGroups
            bool bConsistent = false;
            //遍历spCandidateGroup里的关键帧，判断spCandidateGroup与mvConsistentGroups[iG]是否连续，
            //也就是判断spCandidateGroup和mvConsistentGroups[iG]是否有相同的关键帧
            //FOR3
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                //如果在sPreviousGroup里找到sit
                if(sPreviousGroup.count(*sit))
                {
                    //true表示标记sit所在的spCandidateGroup与sPreviousGroup连续（consistent）
                    //之后要插入到vCurrentConsistentGroups
                    bConsistent=true;
                    //true表示有spCandidateGroup与vCurrentConsistentGroups中的元素存在连续（consistent）
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            //
            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

/**
 * @brief 计算当前帧与闭环帧的Sim3变换等
 *
 * 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与闭环帧的Sim3（当前帧---闭环帧）
 * 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧---闭环帧）
 * 3. 将闭环帧以及闭环帧相连的关键帧的MapPoints与当前帧的点进行匹配（当前帧---闭环帧+相连关键帧）
 *
 * 注意以上匹配的结果均都存在成员变量mvpCurrentMatchedPoints中，
 * 实际的更新步骤见CorrectLoop()步骤3：Start Loop Fusion
 */
bool LoopClosing::ComputeSim3()
{
    /** 为什么需要计算Sim3
     * 当相机从B处开始运动到A处的时候，检测到B为A的闭环候选帧。
     * 此时，考虑到相机从B运动到A的过程中不光会产生旋转和平移的误差，
     * 同时也会产生尺度漂移的累积，需要计算A和B之间的sim3变换，
     * 来找到A和B之间的sim3变换（包括旋转矩阵R、平移向量t、尺度变换s）,
     * 有了这些值之后，就可以对关键帧A的位姿进行纠正
     */

    // For each consistent loop candidate we try to compute a Sim3
    // 对于每一个回环候选关键帧，都计算sim3
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    // 先进行ORB匹配，如果匹配度足够，则开始进行sim3求解
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    //标识nInitialCandidates中哪些keyframe将被抛弃
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    //candidates with enough matches
    //统计满足匹配条件的候选关键帧个数
    int nCandidates=0;

    //遍历候选闭环关键帧nInitialCandidates
    //闭环关键帧与关键帧特征匹配，通过bow加速
    //剔除特征点匹配数少的闭环候选帧
    for(int i=0; i<nInitialCandidates; i++)
    {
        // 步骤1：从筛选的闭环候选帧中取出一帧关键帧pKF
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        // 防止在LocalMapping中KeyFrameCulling函数将此关键帧作为冗余帧剔除
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            //舍弃该帧
            vbDiscarded[i] = true;
            continue;
        }

        // 步骤2：将当前帧mpCurrentKF与闭环候选关键帧pKF匹配
        // 匹配mpCurrentKF与pKF之间的特征点并通过bow加速
        // 返回vvpMapPointMatches[i][j]表示mpCurrentKF的特征点j通过mvpEnoughConsistentCandidates[i]匹配得到的mappoint
        // vvpMapPointMatches[第i个候选关键帧][mpCurrentKF当前帧第j个特征点] = 与当前帧第j个特征点匹配的mappoint
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        // 两帧之间成功匹配特征点数量太少，剔除该候选关键帧
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            //匹配数量足够，准备一个Sim求解器
            //新建一个Sim3Solver对象
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }
        //满足匹配条件的候选关键帧个数+1
        nCandidates++;
    }
    //成功匹配标志，用来跳出while
    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    // 进行RANSAC迭代，对每个候选关键帧
    // 直到某个候选关键帧成功，或者全部都失败
    while(nCandidates>0 && !bMatch)
    {
        //遍历nInitialCandidates
        for(int i=0; i<nInitialCandidates; i++)
        {
            //如果是被丢弃的候选关键帧，则跳过
            if(vbDiscarded[i])
                continue;
            //取候选关键帧
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // 步骤3：对步骤2中有较好的匹配的关键帧求取Sim3变换
            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            // 最多迭代5次，返回的Scm是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）(s,R,t)
            //m代表候选帧pKF，c代表当前帧
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 经过n次循环，每次迭代5次，总共迭代 n*5 次
            // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            // 如果求出了sim3
            if(!Scm.empty())
            {
                //将vvpMapPointMatches[i]中，取inlier(sim3求解得到的)存入vpMapPointMatches
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    // 保存inlier的MapPoint
                    if(vbInliers[j])
                        vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                // 步骤4：通过步骤3求取的Sim3变换引导关键帧匹配弥补步骤2中的漏匹配的mappoint
                // 这里的变换是从相机坐标系2到相机坐标系1的变换
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();

                // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                // 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
                // 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新匹配vpMapPointMatches
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                // 步骤5：Sim3优化，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                //gScm表示候选帧pKF到当前帧mpCurrentKF的Sim3变换
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                // 如果mbFixScale为true，则是6DoFf优化（双目 RGBD），如果是false，则是7DoF优化（单目）
                //vpMapPointMatches表示mpCurrentKF与pKF的mappoint匹配情况
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                // 如果优化结果得到的nInliers满足阈值条件，即求解成功
                if(nInliers>=20)
                {
                    //表示从候选帧中找到了闭环帧 mpMatchedKF
                    bMatch = true;
                    mpMatchedKF = pKF;  //记录这个候选关键帧
                    //gSmw表示世界坐标系到候选帧的Sim3变换
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    //gScm表示候选帧pKF到当前帧mpCurrentKF的Sim3变换
                    //mg2oScw=gScm*gSmw;表示世界坐标系到当前帧mpCurrentKF的Sim3变换
                    mg2oScw = gScm*gSmw;
                    //表示世界坐标系到当前帧mpCurrentKF的变换
                    mScw = Converter::toCvMat(mg2oScw);
                    //记录通过sim3匹配的mappoint
                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    //跳出循环，不再遍历候选关键帧
                    break;
                }
            }
        }
    }

    //如果遍历完所有的候选关键帧，还是没有找到匹配
    if(!bMatch)
    {
        // 清除，返回false
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // 步骤6：
    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // mpMatchedKF: 也就是刚刚匹配成功的候选关键帧
    // 将mpMatchedKF闭环关键帧相连的关键帧全部取出来放入vpLoopConnectedKFs
    // 将vpLoopConnectedKFs的MapPoints取出来放入mvpLoopMapPoints
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();  //获取共视的其他关键帧
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    // 遍历这个闭环关键帧以及它的共视关键帧
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                //将所有共视关键帧以及闭环关键帧的mappoint储存到mvpLoopMapPoints，用来下一步寻找更多匹配
                //不再添加已经添加过的mappoint，因为各个共视关键帧之间会有重复的mappoint
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    // 标记该MapPoint被mpCurrentKF闭环时观测到并添加，避免重复添加
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // 步骤7：将闭环匹配上关键帧以及相连关键帧的MapPoints投影到当前关键帧进行投影匹配
    // Find more matches projecting with the computed Sim3
    /// 目的：
    // 将mvpLoopMapPoints投影到当前关键帧mpCurrentKF进行投影匹配
    // 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
    /// 具体做法:
    // mScw表示世界坐标系到当前帧mpCurrentKF的变换(由sim3得到的)
    // 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，并根据尺度确定一个搜索区域，
    // 根据该MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差小于TH_LOW即匹配成功，更新mvpCurrentMatchedPoints
    // mvpCurrentMatchedPoints将用于SearchAndFuse中检测当前帧MapPoints与匹配的MapPoints是否存在冲突
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // 步骤8：判断当前帧与检测出的所有闭环关键帧是否有足够多的MapPoints匹配
    // If enough matches accept Loop
    // 统计(当前帧)与(闭环关键帧及其共视关键帧)是否有足够多的MapPoints匹配
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    // 足够多，匹配完成
    if(nTotalMatches>=40)
    {
        // 清空mvpEnoughConsistentCandidates，除了匹配上的闭环关键帧
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        //否则，清空全部候选关键帧
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

/**
* @brief 闭环
*
* 1. 通过求解的Sim3以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的MapPoints的位置（相连关键帧---当前帧）
* 2. 将闭环帧以及闭环帧相连的关键帧的MapPoints和与当前帧相连的关键帧的点进行匹配（相连关键帧+当前帧---闭环帧+相连关键帧）
* 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
* 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
* 5. 创建线程进行全局Bundle Adjustment
*/
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 步骤0：请求局部地图停止，防止局部地图线程中InsertKeyFrame函数插入新的关键帧
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    //如果正在进行全局BA，放弃它
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;


        if(mpThreadGBA)
        {
            //1.当使用join()函数时，主调线程阻塞，等待被调线程终止，然后主调线程回收被调线程资源，并继续运行；
            //2.当使用detach()函数时，主调线程继续运行，被调线程驻留后台运行，
            //  主调线程无法再取得该被调线程的控制权。当主调线程结束时，由运行时库负责清理与被调线程相关的资源。
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    // 这里可以利用C++多线程特性修改
    // 等待局部地图停止
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    // 步骤1：根据共视关系更新当前帧与其它关键帧之间的连接
    mpCurrentKF->UpdateConnections();

    // 步骤2：
    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 取出与当前帧在covisibility graph连接的关键帧，包括当前关键帧
    // 取出当前帧和与此当前帧具有连接关系(共视)的关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    //mg2oScw是根据ComputeSim3()算出来的当前关键帧在世界坐标系中的sim3位姿,在ComputeSim3()被设置
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();    //只取位姿的逆， 当前帧相机坐标系的世界坐标系


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        //遍历上面得到的与当前帧共视关键帧
        //步骤2.1：计算mvpCurrentConnectedKFs中各个关键帧（i代表）相对于世界坐标的sim3的位姿，（只是得到，还没有修正）
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            //取关键帧i位姿
            cv::Mat Tiw = pKFi->GetPose();
            //如果不是当前帧, currentKF在前面已经添加
            if(pKFi!=mpCurrentKF)
            {
                //得到从当前帧到关键帧i的变换 Tic
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                //创建sim3，先将尺度赋值为1，变换为上面的Tic，即从当前帧到关键帧i的sim3变换
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                //mg2oScw:当前关键帧在世界坐标系中的sim3位姿(即从世界坐标系到当前关键帧的而变换)，包含尺度因子
                //得到g2oCorrectedSiw : 从世界坐标系到关键帧i的sim3变换，包含尺度因子
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;

                //Pose corrected with the Sim3 of the loop closure
                //关键帧i的sim3修正位姿
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }
            //取关键帧i位姿
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            //g2oSiw 没有使用sim3修正的关键帧i的位姿，尺度为1
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // 步骤2.2：遍历CorrectedSim3(即使用sim3修正位姿之后的关键帧)，修正这些关键帧的MapPoints
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            //取关键帧
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;                //取修正的sim3位姿
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();  //取修正的sim3位姿的逆

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];   //取没有经过sim3修正的位姿，即原来的位姿

            //取关键帧pKFi的mappoint
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            //遍历pKFi的mappoint
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)    //如果当前这个mappoint在下面被标记为已经修正过的
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                // 取这个mappoint原来的坐标
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);    //转换为eigen
                // eigP3Dw:世界坐标系坐标   g2oSiw：还没有使用sim3修正的关键帧i的位姿
                // g2oSiw.map(eigP3Dw): 将世界坐标系使用没有修正的变换转换到关键帧i的相机坐标系
                // g2oCorrectedSwi: 关键帧i修正的sim3位姿的逆，即从关键帧i到世界坐标系的sim3变换
                // eigCorrectedP3Dw: 得到这个mappoint经过sim3修正之后的世界坐标
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                // 保存，更新
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;  //做标记，避免重复修正
                pMPi->mnCorrectedReference = pKFi->mnId;    //表明这个mappoint是使用pKFi这个关键帧来修正的
                pMPi->UpdateNormalAndDepth();   //更新此mappoint参考帧光心到mappoint平均观测方向以及观测距离范围
            }

            // 步骤2.3：将Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            // 将Sim3转换为SE3，更新关键帧的位姿
            // g2oCorrectedSiw：取关键帧i修正的sim3位姿，分别提取出R，t，s
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            // sim3相似变换： p2=s*R*p1+t ===> p2/s=R*p1 + t/s
            // 由于转换为SE3的过程中，直接取了R作为旋转量，那么对应的平移量要除以尺度因子
            // s是一个缩放系数。可以这么理解，p1在旋转后，又被放大了s倍。但是之前的平移量t没有发生变化 <from : https://blog.csdn.net/ziliwangmoe/article/details/90418876>
            eigt *=(1./s); //[R t/s;0 1]

            // R，t拷贝到4x4变换矩阵，
            // QPC: 注意：这个变换矩阵要除以s，即p2=s*R*p1+t ===> p2/s=R*p1 + t/s
            // 为什么要除以s？ 因为在上面的mappoint修正中，有两步，一步是正，一步是反投影，所以最终得到的还是与原来值差不多的
            // 这里只有一步，自然要把尺度还原成1?
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);
            // 关键帧i重新设置位姿
            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            // 更新共视图Covisibility graph,essential graph和spanningtree，以及共视关系
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion    回环融合
        // Update matched map points and replace if duplicated
        // 步骤3：检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
        // mvpCurrentMatchedPoints: 计算sim3的时候，通过sim3匹配的mappoint [这个变量在ComputeSim3()被设置]
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            // 遍历这些成功匹配的mappoint
            if(mvpCurrentMatchedPoints[i])
            {
                //mvpCurrentMatchedPoints[i]=与当前关键帧特征点对应的闭环关键帧的mappoint
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];     //取闭环关键帧mappoint
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);     //取当前关键帧mappoint
                if(pCurMP)  //如果当前关键帧mappoint存在
                    pCurMP->Replace(pLoopMP);       //使用当前关键帧mappoint替换闭环关键帧mappoint
                else
                {
                    //否则
                    //当前关键帧增加mappoint
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    //这个来自闭环关键帧的mappoint增加观测，表示其可以被当前关键帧观测到，对应当前关键帧特征点索引为i
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    //重新计算此mappoint能被看到的特征点中找出最能代表此mappoint的描述子
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // 步骤4：
    // 将闭环关键帧及其邻帧观测到的mappoint投影到当前关键帧进行搜索匹配
    // 目的是：尽量使用闭环关键帧及其共视关键帧所观测到的mappoint来替换旧的mappoint
    // CorrectedSim3: 关键帧i的sim3修正位姿
    SearchAndFuse(CorrectedSim3);

    // 步骤5：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    // 建立新的共视图连接LoopConnections
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    // 步骤5.1：遍历当前帧相连关键帧（一级相连）
    //mvpCurrentConnectedKFs: 当前帧和与此当前帧具有连接关系(共视)的关键帧
    //mappoint融合后，在covisibility graph中，mvpCurrentConnectedKFs附近会出现新的连接
    //遍历和闭环帧相连关键帧mvpCurrentConnectedKFs,只将mvpCurrentConnectedKFs节点与其他帧出现的新连接存入LoopConnections
    //由于这些连接是新的连接，所以在OptimizeEssentialGraph()需要被当做误差项优化
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        // 步骤5.2：得到与当前帧相连关键帧的相连关键帧（二级相连）
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        // 步骤5.3：更新一级相连关键帧pKFi的连接关系，得到新的与当前帧相连关键帧的相连关键帧（二级相连）
        pKFi->UpdateConnections();

        // 步骤5.4：取出该帧更新后的连接关系
        //UpdateConnections()会更新mConnectedKeyFrameWeights(储存着与此关键帧其他关键帧的共视关系及其mappoint共视数量)
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();

        // 步骤5.5：从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        
        // 步骤5.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // 步骤6：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
    // Optimize graph
    // mpMatchedKF：闭环关键帧
    // mpCurrentKF：当前关键帧
    // NonCorrectedSim3：当前关键帧以及共视的关键帧的原位姿
    // CorrectedSim3：当前关键帧以及共视的关键帧的经过sim3修正的位姿
    // LoopConnections: 新的共视连接关系
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // 步骤7：添加当前帧与闭环匹配帧之间的边
    // Add loop edge
    // 添加回环边，在Optimizer::OptimizeEssentialGraph()用来直接添加边
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;

    // 步骤8：新建一个线程用于全局BA优化
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->mnId;
}

/** 目的： 尽量使用闭环关键帧及其共视关键帧所观测到的mappoint来替换旧的mappoint
 * 针对CorrectedPosesMap里的关键帧，mvpLoopMapPoints投影到这个关键帧上与其特征点并进行匹配。
 * 如果匹配成功的特征点本身就有mappoint，就用mvpLoopMapPoints里匹配的点替换，替换下来的mappoint则销毁
 * @param CorrectedPosesMap 表示和当前帧在covisibility相连接的keyframe及其修正的位姿
 */
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    //遍历CorrectedPosesMap
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        //取关键帧i，包含了当前关键帧
        KeyFrame* pKF = mit->first;
        //取关键帧i的sim3修正位姿， 是从世界坐标系到相机坐标系的sim3变换
        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw); //(sR,t)

        //mvpLoopMapPoints: 闭环关键帧及其所有共视关键帧的mappoint
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));

        // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
        ///使用关键帧i的sim3位姿将mvpLoopMapPoints投影到关键帧i，进行匹配搜索,
        ///寻找在关键帧pKF中，与当前这个mappoint可以匹配上的特征点，
        ///使得当前这个mappoint可以和关键帧pKF建立联系
        // mvpLoopMapPoints: 闭环关键帧及其所有共视关键帧的mappoint
        // vpReplacePoint大小与vpPoints一致，储存着被替换下来的mappoint
        // 如果匹配的pKF中的特征点本身有就的匹配mappoint，就用mvpLoopMapPoints替代它。
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            //遍历替换下来的mappoint
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                //将pRep的相关信息继承给mvpLoopMapPoints[i]，修改自己在其他keyframe的信息，并且“自杀”
                //mvpLoopMapPoints[i] : 用来替换的mappoint，也就是新的mappoint
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    //nLoopKF=mpCurrentKF->mnId 当前关键帧id

    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // 更新所有mappoint和关键帧
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    // 如果在全局BA的时候，局部地图器正在激活，这说明可能有新的关键帧插入
    // 新的关键帧没有被包含在这次BA里面，因此通过spanning tree父节点等关系将修正值传播
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped
            // 等待Local Mapping停止
            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            // 从地图的第一个关键帧开始修正关键帧
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                //取一个关键帧
                KeyFrame* pKF = lpKFtoCheck.front();
                //取子节点，共视的关键帧集合
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //取关键帧PKF的位姿的逆，即从相机坐标系到世界坐标系的变换
                cv::Mat Twc = pKF->GetPoseInverse();
                //遍历共视的关键帧集合
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    //nLoopKF=mpCurrentKF->mnId 当前关键帧id
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)    //如果这个值没有被设置，表示是Local Mapping新插进来的关键帧
                    {
                        //如果这个值没有被设置，表示是Local Mapping新插进来的关键帧
                        //需要将修正值传播

                        //Twc: 关键帧PKF的位姿的逆，即从相机坐标系到世界坐标系的变换
                        //Tchildc ： 从关键帧pKF到子关键帧pChild的变换
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        //传播修正到子关键帧的全局位姿
                        //得到世界坐标系到子关键帧pChild的变换，这是经过BA修正的
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;
                    }
                    //表示需要一直传播，直到lpKFtoCheck里面的某个元素(关键帧)没有共视的关键帧集合
                    lpKFtoCheck.push_back(pChild);
                }

                //mTcwBefGBA:储存着BA之前的结果
                pKF->mTcwBefGBA = pKF->GetPose();
                //设置关键帧PKF，就是BA的结果
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            // 修正mappoint
            // 取地图所有mappoint
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;
                //如果这个mappoint没有经历BA,也需要传播修正值
                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    // 表明这个这个mappoint是经过BA的
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    // 这个mappoint没有经历BA

                    //取mappoint参考关键帧,即生成这个mappoint的关键帧为参考关键帧
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    //如果这个mappoint的参考关键帧也没有经过BA，则跳过这个mappoint
                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    // mTcwBefGBA:储存着BA之前的结果
                    // 利用参考关键帧BA之前的位姿，将mappoint投影到参考关键帧相机坐标系
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    // 使用参考关键帧BA之后的结果，将mappoint反投影到世界坐标系
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);
                    // 设置修正的坐标
                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }
            //表明地图有大改变
            mpMap->InformNewBigChange();
            //
            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }
        //没用的标志
        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
