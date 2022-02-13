#pragma once
class SysInfo  {
        public:
        static SysInfo& instance ()     ;  
         ~SysInfo ()     ;  
        virtual void init ()  =0   ;  
        virtual double cpuLoadAverage ()  =0   ;  
        virtual double memoryUsed ()  =0   ;  
        protected:
         SysInfo ()     ;  
        private:
         SysInfo (const SysInfo& rhs)     ;  
        SysInfo& operator= (const SysInfo& rhs)     ;  
};
