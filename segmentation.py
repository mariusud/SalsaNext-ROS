import rospy
import message_filters
import torch
import yaml
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from common.laserscan import LaserScan, SemLaserScan
from models.SalsaNext import *
from postproc.KNN import *
import time
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from std_msgs.msg import Header
from common.laserscanvis import LaserScanVis

class SegmentationNode():
    def __init__(self, knn=True, modeldir='models/',
                arch_cfg = 'models/arch_cfg.yaml', data_cfg = 'models/data_cfg.yaml'):
        self.uncertainty = False
        self.max_points =  150000
        self.node = rospy.init_node('segmentation_node')

        self._init_pcl()

        self.pub = rospy.Publisher("/colored_points", PointCloud2, queue_size=10)

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
        self.sensor = ARCH["dataset"]["sensor"],
        self.sensor = self.sensor[0] #weird but works
        self.sensor_img_means = torch.tensor(self.sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(self.sensor["img_stds"],
                                        dtype=torch.float)
        # Load DL model 
        self.model = SalsaNext(n_classes)
        #self.model = nn.DataParallel(self.model)
        w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
        tups = self.model.load_state_dict(w_dict['state_dict'], strict=True)
        print(tups)
        print(self.model)
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

        rospy.Subscriber("/lidar/points", PointCloud2, self.infer)
        rospy.spin()

    def project_scan(self, data):
        # self.scan = LaserScan(project=True,
        #                H=int(self.sensor["img_prop"]["height"]),
        #                W=int(self.sensor["img_prop"]["width"]),
        #                fov_up=int(self.sensor["fov_up"]),
        #                fov_down=int(self.sensor["fov_down"])
        #                 )

        self.scan = SemLaserScan(project=True,
                       H=int(self.sensor["img_prop"]["height"]),
                       W=int(self.sensor["img_prop"]["width"]),
                       fov_up=int(self.sensor["fov_up"]),
                       fov_down=int(self.sensor["fov_down"])
                       )
        # open and obtain scan
        raw_cloud = pc2.read_points_list(data,skip_nans=True,field_names=("x", "y", "z", "intensity"))
        cloud = list(filter(lambda num: not math.isinf(num[0]), raw_cloud))
        self.scan.open_ROS_scan(cloud)
        #self.write_cloud(cloud)

        itemindex = np.where((self.scan.proj_y > 0) & (self.scan.proj_x > 0))[0]
        #self.cloud = np.take(cloud,itemindex,axis=0)
        #self.cloud = np.delete(self.cloud,3,axis=1)

        #print("clouding")
        #fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #          PointField('y', 4, PointField.FLOAT32, 1),
        #          PointField('z', 8, PointField.FLOAT32, 1),
        #          ]
        #pcp = pc2.create_cloud(self.header, fields, self.scan.points)
        #self.publish_pcl(pcp)


        self.scan.proj_y = np.take(self.scan.proj_y,itemindex,axis=0)
        self.scan.proj_x = np.take(self.scan.proj_x,itemindex,axis=0)
        self.scan.points = np.take(self.scan.points,itemindex,axis=0)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = self.scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(self.scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(self.scan.unproj_range)
        #unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        #unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(self.scan.proj_range).clone()
        proj_xyz = torch.from_numpy(self.scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(self.scan.proj_remission).clone()
        proj_mask = torch.from_numpy(self.scan.proj_mask)

        proj_labels = []

        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(self.scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(self.scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()
        return proj, proj_mask, proj_x, proj_y, proj_range, unproj_range, unproj_n_points, self.scan.selected_points


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
            
        proj_in, proj_mask, p_x, p_y, proj_range, unproj_range, npoints, selected_points = self.project_scan(data)
        # first cut to rela size (batch size one allows it)
        #p_x = p_x[0, :npoints]
        #p_y = p_y[0, :npoints]
        #proj_range = proj_range[0, :npoints]
        #unproj_range = unproj_range[0, :npoints]
        self.gpu = False
        if self.gpu:
            proj_in = proj_in.cuda()
        #p_x = p_x.cuda()
        #p_y = p_y.cuda()
        #if self.post:
            #proj_range = proj_range.cuda()
            #unproj_range = unproj_range.cuda()
        if not self.gpu:
            proj_in = torch.unsqueeze(proj_in,0)
        proj_output = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network in", res, "sec")
        end = time.time()
        #cnn.append(res)

        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()
        #res = time.time() - end
        #print("Network seq", path_seq, "scan", path_name,
        #        "in", res, "sec")
        #end = time.time()
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
        print("KNN Infered seq in", res, "sec")
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        unproj_argmax = proj_argmax
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        self.scan.set_label(pred_np)
        labs = np.reshape(self.scan.sem_label,(self.scan.sem_label.shape[0],1))

        pc =  np.append(self.scan.points, labs,axis=1)
        pc =  np.append(self.scan.points, labs,axis=1)

        pcp = pc2.create_cloud(self.header, self.fields, pc)
        self.publish_pcl(pcp)

        #vis = LaserScanVis(scan=self.scan,
        #            scan_names="",
        #            label_names="",
        #            offset=1,
        #            semantics=True,
        #            instances=False)
        #vis.run()

        # map to original label
        #pred_np = to_orig_fn(pred_np)

        # save scan
        #path = os.path.join(self.logdir, "sequences",
        #                    path_seq, "predictions", path_name)
        pred_np.tofile("pred.label")
        #return pred_np'
        rospy.signal_shutdown('s') 

    def publish_pcl(self,pcp):
            pcp.header.stamp = rospy.Time.now()
            print("publishing")
            self.pub.publish(pcp)

    def _init_pcl(self):
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  #PointField('intensity', 12, PointField.FLOAT32, 1),
                  PointField('label', 20, PointField.FLOAT32, 1),
                  ]
        self.header = Header()
        self.header.frame_id = "vehicle_blue/lidar_link/gpu_lidar" #add eventually world ?

    def write_cloud(self, cloud):
        readcloud = pc2.read_points_list(cloud, skip_nans=True, field_names=("x", "y", "z", "intensity", "label"))
        N = len(readcloud)
        arr = np.zeros((N,4),dtype=np.float32)
        label = np.zeros((N,1),dtype=np.float32)
        for n, point in enumerate(readcloud):
            arr[n,0] = point[0] #might be different xyz
            arr[n,1] = point[1] #might be different xyz
            arr[n,2] = point[2] #might be different xyz
            arr[n,3] = point[3] # reflectivity
            label[n] = point[4] #might be different xyz
        arr.astype('float32').tofile('1.bin') # add location
        label.astype('float32').tofile('1.label') # add location
        self.name_incrementer += 1


if __name__ == '__main__':
    node = SegmentationNode()