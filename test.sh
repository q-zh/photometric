for i in $( seq 1 10)
do
	python eval/run_stage2.py --test_set ${i} --test_save ${i} --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/checkp_20.pth.tar --retrain_s2 data/logdir/UPS_Synth_Dataset/TPAMI/10_images/checkp_10.pth.tar
done