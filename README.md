# MTCNN-FaceNet
《人脸检测开发》课程项目——电视剧《都挺好》第一集人脸统计识别:

针对给定视频——《都挺好》第一集，统计该视频中出现过的人物共多少位。使用课程中讲授的[MCTNN](https://github.com/wangbm/MTCNN-Tensorflow)和[facenet](https://github.com/davidsandberg/facenet)两个模型，使用MTCNN检测视频中的人脸，使FaceNet进行人脸识别。

- 使用MTCNN进行人脸检测：
  ``` Python
  python detect_face.py "video.mp4" --save_path './datasets/mtcnn_160_face/video_img/' --capture_interval 24 
  ``` 
  默认帧间隔为24，检测结果存放在"./datasets/mtcnn_160_face/"下

- datasets中已经有了检测库和待识别人脸库，可以跳过这步，直接进行人脸识别

- facenet预训练模型：

  '20180408-102900/20180408-102900.pb' （CASIA-WebFace）

  '20190218-164145/20190218-164145.pb'（基于亚洲人脸）

  默认使用官方基于CASIA-WebFace训练的模型

- 人脸识别（欧式距离）：
  ``` Python
  python recognition.py "./datasets/mtcnn_160_face/img/" --dist 1.1 --split_img True --save_path './video_face_result/'
  ``` 
  `split_img` 默认为`False`,设置为`True`时才会调用split.py对待识别人脸进行模糊检测，在原人脸库相同路径下产生两个新的待识别人脸库（clear&blurry）

  `dist`默认距离阈值1.1，结果人脸库在"./video_face_result/"下

- 人脸识别（余弦相似度）：
  ``` Python
  python recognition-cos.py "./datasets/mtcnn_160_face/img/" --dist 1.1 --split_img True --save_path './video_face_result/'
  ``` 
  结果人脸库在"./video_face_result/cos/"下

- 检测识别同时进行（欧式距离）：
  ``` Python
  python detect_reco_all.py "video.mp4" 
  ``` 

  默认距离阈值1.1，检测和识别结果存放在"./datasets/mtcnn_160/"下

