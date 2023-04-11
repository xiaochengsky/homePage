const langObj = {
  zh: {
    name: '杨成',
    lang: 'Eng',
    news: '近况',
//    cv: `我现就职于深圳某国企单位，从事深度学习和图像处理相关的工作和研究。
//        在此前，我于2021年6年在华中科技大学获得电子与通信工程硕士学位，于2018年在湖南农业大学获得电子信息工程学士学位。`,
    cv: `我于2021年6年在华中科技大学获得电子与通信工程硕士学位，于2018年在湖南农业大学获得电子信息工程学士学位。`,
    experience: '经历',
    my_experience: '较为了解嵌入式开发，软件开发，计算机视觉等领域，具体履历如下: ',

    yolov5: `
        在模型部署阶段，不得不在精度和参数量上做一个权衡。其实在模型中有许多参数是冗余的，去掉它们并不会对模型精度造成太大的影响。
        这里我以业内公认的高精度、低参数量的yolov5作为baseline，coco数据集作为测试数据，调研了近几年主流的6种模型剪枝和3种蒸馏的方法。
        最终得到的tiny-yolov5在保持mAP精度不变的情况下，去除了35%参数量，cpu推理增速约25%+。`,

    general: `
        基于我开发的用于图像分类和检索pipeline-PyCR和通用yolov5检测器，我解决了多种分类问题（城市交通：树木倒伏、植被破坏、路面标牌、20余种
        车辆的分类）任务，以及多种检测任务（火焰、烟雾、扬尘），并且都已经上线使用。基于这些pipeline，可以很方便的对相似任务做快速迭代和部署。`,

    yitu: `
        该任务主要是在样本极度不均衡的情况下优化ReID场景下的分类模型。其中大约骑车人3w，行人300w的数据量，需要优化出可靠的分类模型并以此作为
        topic。在调研了GAN、图像合成等领域后，在以分割、姿态检测为基础模型的基础上，提出了基于CPP的方法。该方法能改善ReID场景下，行人在不同
        状态下的检索问题，具体请查阅Arixv。`,

    baidu: `
        商品自动结算台是一个具体高fps(在云部署的情况下能1s得到结果)、高账单精度(98%+)的产品。它基于目标检测+图像检索方法实现。
        在这个项目中，我负责图像处理部分。采用目标检测区分前背景，以此来降低检测难度，同时采用图像检索代替分类，来增强类别识别的鲁棒性，以及避免
        随着商品的小数量级别的加入而需要频繁的切换模型的痛点问题。
        在此项目中，我使用SPP、DCN、IOU Aware来改进yolov3模型，采用ResNeXt101作为分类模型，并从其Pool5层得到feature作为样本的特征向量，
        用于后续检索。为了提升检索精度，采用二阶段双loss（Triplet+Softmax）训练。`,

    tx1: `基于Golang开发出跨平台（Win, Linux, Mac）的腾讯云音视频AI服务的中继转发接口。保证高可用性和高QPS。`,

    tx2: `宙斯盾-蓝军对抗平台用于多方位验证腾讯安全系统的稳定性。采用多主多从机方式接受用户请求和调用不同的DDOS攻击确保平台的性能和功能。
        这其中需要完成web界面和后端系统的开发。`,

    lifx: `这是 Lifx 灯泡球项目，通过udp组网实现手机对多个Lifx灯泡球的配置（web protal 配置）和控制（开关、色温、亮度）。
          在此项目中，我承担部分硬件电路PCB的设计以及ESP8266作为MCU的驱动程序的开发，采用基于Lifx protocol（基于udp）的开发协议。后续我也
          使用Docker搭建出基于ESP8266的开发工具链。
          值得一提的是，Lifx 灯泡球确实很cool。因为这是在我本科阶段完成的第一个商业项目，并且现在仍在售卖中。`,

    huawei: `这是华为2020年全球AI竞赛中图像处理赛道的任务，任务是电子产品的图像检索，评价指标是mAP@10。任务的难点特性有宽细粒度范围的图像检索，
    例如有需要对华为手机和苹果手机进行区分的高细粒度特征辨识需求，也有需要对显示屏和充电器进行区别的粗细粒度特征识别需求。我们（队伍共两人）
    使用一张2080Ti，最终获得16名。我们的一些参赛细节如下： <br>
    1) 选择基础的分类网络模型，例如 EfficientNet-B5, SEResnext101, ResNeSt50等; <br>
    2) 数据5折预处理和增强训练，交叉验证，等等; <br>
    3) 特征预处理: TTA, PCA 白化, 特征聚合, re-rank重排; <br>
    4) 错误难样本分析, 我们发现有100余例多主题场景下的图像（例如鼠标、显示屏、机箱在一个图像中，最后label是显示屏）等等。这导致在做统一pool的
    时候容易混淆特征，所以采用CAM激活图做聚类分割主体的方法，分别获取主体的特征并做聚合处理。 <br>`,

    kaggle: `这是关于kaggle小麦头的检测任务，有包括我在内的两人队伍完成，测评指标是mAP[0.5:0.75::0.05], 相比于一般的AP公式，这里的AP会
    考虑召回率：AP = TP / (TP + FP + FN)，我们使用单卡2080Ti完成： <br>
    1) 选择yolov5作为基础模型； <br>
    2) 数据预处理，anchor重聚类, mosaic等多种增强，以及对权重进行EMA处理； <br>
    3) 简单调参，grid search寻找左右 iou threshold; <br>
    4) 单模型后处理：TTA + WBF + PLabel + OOF; <br>
    5) 错误难样本分析: 分析模型对于通用的遮挡场景下小麦头的bbox的重叠问题，手动添加策略调整; <br>
    6) 最后虽然我们取得了top2%的成绩，但是由于yolov5 git 只是GPL协议，所以导致成绩被取消。但是这一次的竞赛经历为我日后形成自己的竞赛处理思维
    建立了比较规整的范式。`,

    proj: `
        在我硕士生涯前半段(2018/09~2019/12)，我主要从事硬件编程，嵌入式开发，软件开发相关的任务，同时我完成了我的毕业论文和所有实验。
        在后半段（2020/01~2021/06），我开始主攻计算机视觉领域，并以它作为我的研究方向。`,

    pycr: `
        当我完成了竞赛后，我思考如何建立一套属于自己的能高效迭代的pipeline专门用于图像的分类和检索任务。
        在此前的调研中，我并没有发现有兼容这两部分的框架。例如京东的FastReID是专门针对检索任务提出的框架，旷视的PyRetri也是如此，并且它不支持
        模型训练，只是专门针对检索任务的后处理部分（而且代码有内存泄露…）。所以在参考它们这些优秀组件的基础上，我着手于搭建适用于我自己的图像分类
        和检索pipeline-PyCR(Pytorch for Classification and Retrieval)。`,

    football: `
        这不仅是一个研究课题（我的硕士毕业设计），而且是一个工程项目。需要设计一个低复杂度、高效率（接入量）的足球运动员检测系统。主要用于监测
        运动员的生理信息，例如运动速度、心率、实时位置、跳跃次数和高度等等。<br>
        这个项目分为三大部分，分为是通信节点（30+个）、网关（1个）、显示上位机（1个）。所有的通信节点佩戴在运动员身上，用于收集生理数据。网关作
        为服务端，由STM32、CC1310x2、ESP8266和其它外设组成。网关负责上位机和标签节点的指令和数据通信。上位机终端用于指令下发和各个节点的数据
        展示。其中STM32用于整体逻辑控制，两个CC1310用于射频收发与通信节点交互，ESP8266充当AP+STA用于上位机wifi接入。每个部分都采用有限
        状态机（FSM）实现，以保证任务驱动和执行的稳定。总体设计成本（300$）、高接入数量（>30）、长距离通信（>150m）、低功耗（<2mA）。我完成了
        整个系统的硬件、软件、通信协议和相应的优化算法，具体可查阅我的<a id="attach" target="_blank" title="title" href="./paper.pdf">硕士论文</a>。<br>
        同时，我感谢刘师兄帮助我在高频电路上的设计，感谢我同学帮我采集数据、做实验，处理部分bug，感谢我导师郭老师对我长期的指导。`,

  },

  en: {
    name: 'YANG Cheng',
    lang: '中文',
    news: 'news',
    cv: `
            I am a machine vision engineer at a state-owned enterprise, Shenzhen,
            where I work on deep learning and computer vision, etc.             Before that, I did my master degree at school of EIC, <a href="https://www.hust.edu.cn/">Huazhong University of Science and Technology(HUST) </a>, Wuhan, China, in 2021,             where I was advised by Prof. <a href="http://eic.hust.edu.cn/professor/guopeng/">Peng Guo</a></a>.             I did my bachelors at school of Information and Intelligence, <a href="https://www.hunau.edu.cn/">Hunan Agricultural University(HUNAU) </a>, Changsha, China, in 2018.`,
    experience: 'Experience',

    my_experience: `
            I am curious about many things when I am a student. Also I wanted to find a research direction that suits me. During undergraduate to master study,
            I have learned <strong>embedded development</strong>, <strong>software development</strong>, <strong>computer vision</strong> and other related technologies.
            My experiences and projects are as follows.`,

    yolov5: `
        When we deploy neural network models, we have to face their high computational cost and memory footprint. It seems that we have to make a strict choice
        between hardware cost and accuracy. But we know almost all models have many redundent parameters. I summarized 6 methods of model pruning and 3 methods of model knowledge distillation as mainly benchmarks in YoLov5.
        In short, I implemented reduce FLOPs and Parameters by about 35% without negatively impacting the mAP by using those methods. The detailed benckmarks will coming.`,

    general: `
        Based on my <a href="https://github.com/xiaochengsky/PyCR" style="color: #447ec9" <strong><font size="2"> PyCR(Pytorch for Classification and Retrieval) </font></strong></a> pipeline and YoLov5 detector,
        I solved some projects about classification(Fallen trees, Vegetation destruction with <i>weak supervision</i>, road slogan, 20+ types of vehicles and so on) and detection(general traffic, fire, fumes, dust) tasks.
        By using it, I can achieve high accuracy(98+%) easily and quickly in these projects.`,

    yitu: `
        Those project aim to optimize some classification models, such as <i><strong>"Cyclist and pedestrian classification model"</strong></i>  <br>
        In this projects of ReID, I only used 3w imbalanced( <i><strong>cyclist: pedestrain ~ 1:30</strong></i>) images to complete the classification of 300w images. Through research related work, such as gan, image composition,
        I proposed a method callde '<strong>Copy and Paste Based on Pose(CPP)</strong>', which can effectively alleviate the sample imbalance problem in this situation.
        Then I used a simple model(EfficientNetB5) and corresponding training skills to achieve classification accuracy of 99.6%. For detailed information about <i><strong>CPP</strong></i>, please see
        <a href="https://arxiv.org/abs/2107.10479">arXiv</a> for more detail information.`,

    baidu: `
            This project aims to achieve high-precision commodity bill settlement. When uses place the dinner plate under its camera, it will quicky automatically
        generate the correct bill within 1 second. <br>
        For me, I was responsible for its visual processing, such as commodity detection and classification. In order to make it work more stable, I combined object detection and retrieval to solve the problem and avoid
        frequently updates of the model. Specifically, I decoupled the problem of commodity detection into two sub-problems of the object detection and
        object retrieval. The detection model is responsible for detecting foreground object, and the retrieval model is responsible for identifying the
            category of the object. The final overall accuracy can reach <i><strong>98%+</strong></i>. <br>
        I used <strong><i>YoLov3</i></strong> as the basic detection model in this project. By following the progress at that time, and making
        trade-offs in inference time and accuracy, I added <i><strong>SPP, DCN, IOU Aware</strong></i> to <i><strong>YoLov3</strong></i> to improve
        detection accuracy. In addition to this, I used <i><strong>ResNeXt101</strong></i> as the basic classification model. and the output from <i><strong>
        pool5</strong></i> is used as the metric vector during retrieval. To improve the accuracy of retrieval, I add double loss(<i><strong>Softmax loss, Triplet loss</strong></i>) to restrict feature space, so that lead to the model output with a larger inter-class
        variation and a smaller intra-class variation.`,

    tx1: `        This project aims to design a cross-platform image and speech processing interface tool based on Golang. Because
        for the convenience of users to use some image and speech processing services, I wanted to design a cross-platform
        tools, which allowed uers to experience many basic services that deployed on cloud servers, no matter what operating system they used(Windows, Linux, Mac).
        And These services include image recognition, image detection, speech recognition, ocr, etc.`,

    tx2: `        This project aims to design a attack platform called <i><strong>Aegis—The Blue Army Attack Platform</strong></i>, which used to verify the
        performance of the company's defense platform via some DDOS attacks. <br>
        In this project, I was responsible for the frontend and backend programs. When users input the attack request with specific parameters to web,
        the web server will forward the request to the corresponding server. Then, the server as a Zombie computer to call the
        corresponding script to attack target machine. And the server reflects the attack status in real-time to the web for display to users. Last but
        not least, in order to maintain control of the attack state and ensure the reliability of the system, even if the corresponding server suddenly outage,
        I must keep in sync between different servers.`,

    lifx: `        This product called the <i><strong>LIFX blub</strong></i>. We can control one or more bulbs by using mobile phone's wifi, and there was a variety of
        lighting effects. It was supported real-time control, flexible portal configuration in mobile, and so on.<br>
        In this project, I was responsible for the its hardware program control. We used the ESP8266 as the Micro Controller Unit to drive bulbs, which comes with a
        lightweight network protocol stack and can act as Access Point or Staion. We adopted the <a style="color: #447ec9" href="https://lan.developer.lifx.com/docs/introduction">lifx protocol</a> which
        based on the UDP network protocol. so, Based on the LIFX protocol and reliable program, we have implemented many functions by only using a moblie
        phone to control it. For example, Registering and resetting the network configuration for web's portal configuration, setting a variety of light
        colors and intensities, and driving multiple bulbs under the same subnet at the same time by the UDP broadcast. <br>
        BTW, I would like to say, the <span style="color:red">LIFX Blub is pretty cool!</span>. I mean, for me, it's my first commercial project at the undergraduate level, and it’s on sale.`,

    huawei: `      This is an image retrieval task about electronic products, and we(2 people in the team) need to find 10 electronic products that are most similar to each query dataset from gallery dataset. It has a wide range of
      fine-gained requirements. For example, sometimes it need to distinguish between different types of mobile phones(Huawei and apple), and sometimes, it need to distinguish between different
      products (display screens and chargers). The metric is (0.5 * top1 + 0.5 * mAP@10). We only used one 2080Ti GPU: <br>
      step 1) Choosing three basic network as ours models: EfficientNet-B5, SEResnext101, ResNeSt50; <br>
      step 2) Data preprocessing and model training: 5-fold cross validation, albumentations, warm up, label smoothing, Adam, GeM, BN Head, ArcFace, EMA and so on; <br>
      step 3) Feature post-processing: TTA, PCA Whitening, feature fusion, and Re-rank. <br>
      step 4) Analyzing some badcases: We found that there are multi-objects scenes in the datasets(About 100+). For example, in an image, there are both a display, a host and a mouse, and they lead to unclear feature
      representation. Because we were surprised to find our model cannot obtain similar results in these images. So, we used their attention map to cut the image into multiple small images corresponding to each subject,
      then we aggregated the feature of multiple small images as its features.`,

    kaggle: `
          This is an object detection task about Wheat Detection, and we(2 people in the team) need to mark all the wheat heads in echo images. The metric is mAP@[.5:.75], and the AP = TP / (TP + FP + FN). We only used one 2080Ti GPU<br>
      step 1) Choosing the yolov5 as ours basic model; <br>
      setp 2) Data preprocessing, anchor re-clustering and model training: the Mosaic and albumentations, label smoothing, EMA as so on; <br>
      step 3) Simple parameter adjustment: find a better Iou threshold when calculating nms; <br>
      step 4) Post-processing: TTA + WBF + PL + OOF; <br>
      step 5) Analyzing some badcases: We found that there are some redundant prediction bounding bbox when the wheat occlusion is serious. So we summarized the corresponding rules and peform filtering. <br>
      In conclusion, It was the first time I participated in the competition. Some time after that, I re-summarized the competition, and found some error methods, such as only adjusting iou threshold is the wrong way, analyzing
      badcase and setting some rules to 'solov' the misdetection will undoubtedly only lead to over-fitting results. This experience taught me that when analyzing problems, we must learn to consider from a macro
      perspective. At the same time, the analysis of badcase should not be discussed separately from model and loss. <br>
      BTW, The controversy caused by yolov5 is too great, our results is were cancelled. If not, our score should be in the top 2%.`,

    proj: `
        In the second half of the master's career(01/2020 ~ 06/2021), I mainly focus on computer vision, and take it as my research direction.
        In the first half of the master's career(09/2018 ~ 12/2019), I mainly focus on hardware programming, embedded development, software development, and have learned some hardware-related knowledge. At the same time, my master's dissertation is
        also closely related to them.`,

    pycr: `
        When I finished the HuaWei's competition, I started to think about how to complete some tasks efficiently. The classification and retrieval are familiar to me, So I was thinking about how to build a
        pipeline, which can make me implement these functions better and faster. <br>
        At this stage, there is no pipeline that directly merge classification tasks and retrieve tasks. For example, FastReID is a framework for end-to-end retrieving in ReID, PyRetri is a pipeline that directly
        uses features for post-processing, and does not involve the training process. So, I have build a pipeline, which can do the end-to-end processing of classification and retrieval respectively.
        For detailed processing methods, please see <a style="color: #447ec9" href="https://github.com/xiaochengsky/PyCR">github</a>.`,

    football: `
            This is research topic, but also a project. Its goal is to design a low-complexity, low cost and high-efficientcy(QPS) product which serves group sports monitoring. It's mainly to monitor some
        pyhsical information of athletes, such as speed, heart rate, real-time position, number of jumps and height, etc. <br>
        The projects is divided into three parts, which include a lot of collection nodes(30+), the gateway(1), and display terminal(1). Each collection nodes is composed of CC1310 and a variety of sensors, which are
        used to collect real-time physical data of athletes. The gateway is mainly composed of STM32, multiple CC1310, ESP8266 and other auxiliary chips(FLASH, Power management). The function of the gateway is
        is responsible for transmission of data from multiple nodes to the display terminal and distribute commands from the terminal to all nodes. The display terminal is responsible for some command control and
        visualization of data from the multiple nodes. Each parts is implemented based on the corresponding Finite State Machine.
        To guarantee the stability of the system, low complexity(~$300), high access volume(~1Hz), long-distance communication(>150m) and low power consumption(<2mA) functions, I have completed the corresponding hardware, software and
        communication and data processing algorithm design. Please see my <a id="attach" target="_blank" title="title" href="./paper.pdf">master's dissertation</a> for the detailed informations. <br>
        Last but no least, Although this is my master's dissertation, I would like to think my senior Liu for his helped in high-frequency circuit design, and thank my classmates for testing with me and helping me find bugs,
        and thank my supervisor Peng Guo for his long-term guidance.`,

  },
}
/**
 *  [[className1,content1],[className2,content2],...]
 *  className对应类名，一般情况下是唯一值
 *  content对应langObj的子项中的key的值
 */
const allElement = [
  ['my-name', 'name'],
  ['my-cv', 'cv'],
  ['lang', 'lang'],
  ['news', 'news'],
  ['experience', 'experience'],
  ['my_experience', 'my_experience'],
  ['yolov5', 'yolov5'],
  ['general', 'general'],
  ['yitu', 'yitu'],
  ['baidu', 'baidu'],
  ['tx1', 'tx1'],
  ['tx2', 'tx2'],
  ['lifx', 'lifx'],
  ['huawei', 'huawei'],
  ['kaggle', 'kaggle'],
  ['proj', 'proj'],
  ['pycr', 'pycr'],
  ['football', 'football'],
]
