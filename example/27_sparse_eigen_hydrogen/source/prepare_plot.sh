./vis_00_cuda_main | tee o
cat o |grep Eigenvec|cut -d "=" -f 3,4|cut -d "'" -f 2,4|tr "'" " " > o2
