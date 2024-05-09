#ifndef VIS_07_JOB_H
#define VIS_07_JOB_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Job {
public:
  virtual void Main();

protected:
  friend class JobThread;
  void RunCodeWrapper();
};
class JobThread {
public:
  void CreateAndStartThread(unsigned int threadId);
  void WaitForThreadToStop();
  void Go();
  void BackgroundTask();
  HANDLE m_GoSignal, m_ThreadHandle;
  int m_ThreadID;
};
class JobManager {
  void ~JobManager();
  static void CreateJobManager(unsigned int numThreads);
  static JobManager *GetJobManager();
  static void GetProcessorCount(uint &cores, uint &logical);
  void AddJob2(Job *a_Job);
  unsigned int GetNumThreads();
  void RunJobs();
  void ThreadDone(unsigned int n);
  void MaxConcurrent();

protected:
  friend class JobThread;
  Job *GetNextJob();
  Job *FindNextJob();
  static JobManager *m_JobManager;
  Job *m_JobList[256];
  CRITICAL_SECTION m_CS;
  HANDLE m_ThreadDone[64];
  unsigned int m_NumThreads, m_JobCount;
  JobThread *m_JobThreadList;
};
#endif