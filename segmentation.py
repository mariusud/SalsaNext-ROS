import rospy
import message_filters
import torch
import yaml
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from common.laserscan import LaserScan
from models.SalsaNext import *
from postproc.KNN import *

class SegmentationNode():
    def __init__(self, knn=True, modeldir='models/',
                arch_cfg = 'models/arch_cfg.yaml', data_cfg = 'models/data_cfg.yaml'):
        self.uncertainty = False
        self.node = rospy.init_node('segmentation_node')

        #self.pcl_cb = message_filters.Subscriber("/lidar/points", PointCloud2)

        rospy.Subscriber("/lidar/points", PointCloud2, self.infer)
        # open arch config file
        try:
            print("Opening arch config file %s" % arch_cfg)
            ARCH = yaml.safe_load(open(arch_cfg, 'r'))
        except Exception as e:
            print(e)
            print("Error opening arch yaml file.")
            quit()
    
        # open data config file
        try:
            print("Opening data config file %s" % data_cfg)
            DATA = yaml.safe_load(open(data_cfg, 'r'))
        except Exception as e:
            print(e)
            print("Error opening data yaml file.")
            quit()
        n_classes = len(DATA["learning_map_inv"])

        # Load DL model
        self.model = SalsaNext(n_classes)
        #self.model = nn.DataParallel(self.model)
        w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)


        if knn:
            self.post = KNN(ARCH["post"]["KNN"]["params"], n_classes)


        # GPU
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
          cudnn.benchmark = True
          cudnn.fastest = True
          self.gpu = True
          self.model.cuda()

    def project_scan(self, data):
        scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down
                        )
        # open and obtain scan
        raw_cloud = pc2.read_points_list(data,skip_nans=True,field_names=("x", "y", "z", "intensity"))
        cloud = list(filter(lambda num: not math.isinf(num[0]), raw_cloud))
        scan.open_ROS_scan(cloud)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)

        proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        # return
        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points


    def infer(self, data): #to_orig_fn,cnn,knn):
        # switch to evaluate mode
        if not self.uncertainty:
          self.model.eval()
        total_time=0
        total_frames=0
        # empty the cache to infer in high res
        if self.gpu:
          torch.cuda.empty_cache()

        with torch.no_grad():
          end = time.time()
            

        proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints = self.project_scan(data)

        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
            proj_in = proj_in.cuda()
        p_x = p_x.cuda()
        p_y = p_y.cuda()
        if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        
        proj_output = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
                "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
                "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                        unproj_range,
                                        proj_argmax,
                                        p_x,
                                        p_y)
        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
                "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)



if __name__ == '__main__':
    node = SegmentationNode()