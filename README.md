# fpdw4win
This is the windows version of FPDW-https://github.com/apennisi/fastestpedestriandetectorinthewest

用 caffe 做回归 （上）
发表于2016/11/12 21:13:18  941人阅读
分类： 深度学习

最近项目需要用到caffe来做关键点的回归，即通过caffe来训练一个网络，输出的结果不是简单地类别，而是一些坐标（浮点数）。
下面的这篇博文对caffe做回归有一个比较好的介绍：
http://www.cnblogs.com/frombeijingwithlove/p/5314042.html
这篇博文使用的是HDF5+python的方式。而我采用的是直接修改caffe的.cpp文件，并重新编译的方式，两种方式各有利弊，我个人认为理解并修改源码对进一步理解caffe很有帮助。当然配置了faster-rcnn或者SSD之后也可以做回归。
caffe本来就“擅长”于做分类任务，所以要拿caffe来做回归任务，就需要对caffe的源码做一些修改。修改的地方主要是下面两大部分：
1、 制作lmdb文件相关的代码（即修改convert_imageset.cpp文件）：image to Datum
2、 读取lmdb文件相关代码（即修改data_layer.cpp文件）：Datum to Blob
根据这两大部分，我将博文分为上下两篇，本文为上篇，关于如何制作用于回归的lmdb文件。
首先，看一看用于分类的txt文件：
cat_1.jpg 0
cat_2.jpg 0
dog_1.jpg 1
dog_2.jpg 1
 里面是图片的名称以及对应的类别（这里不考虑多标签的情况）。
而用于做关键点回归的txt文件：
cat_1.jpg 0.03 0.45 0.55 0.66
cat_2.jpg 0.44 0.31 0.05 0.34
dog_1.jpg 0.67 0.25 0.79 0.56
dog_2.jpg 0.89 0.46 0.91 0.38
 后面带有多个归一化的坐标（上面的是我随便举的例子，没有实际的意义），实际应用中它们可能代表着某一个BoundingBox的坐标，或者是脸部一些关键点的坐标。
下面我将一一列出需要修改代码的地方，带有//###标记的就是我修改的地方：
首先是对tools/convert_imageset.cpp进行修改，复制tools/convert_imageset.cpp，并重新命名，这里姑且命名为convert_imageset_regression.cpp，依然放在tools文件夹下面。
// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <boost/tokenizer.hpp> //### To use tokenizer
#include <iostream> //###

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

using namespace std;  //###

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  
  std::ifstream infile(argv[2]);
  //std::vector<std::pair<std::string, int> > lines;  //###
  std::vector<std::pair<std::string, std::vector<float> > > lines;
  std::string line;
  //size_t pos;
  //int label;  //###
  std::vector<float> labels;

  while (std::getline(infile, line)) {
    // pos = line.find_last_of(' ');
    // label = atoi(line.substr(pos + 1).c_str());
    // lines.push_back(std::make_pair(line.substr(0, pos), label));
    //###
    std::vector<std::string> tokens;
    boost::char_separator<char> sep(" ");
    boost::tokenizer<boost::char_separator<char> > tok(line, sep);
    tokens.clear();
    std::copy(tok.begin(), tok.end(), std::back_inserter(tokens));  

    for (int i = 1; i < tokens.size(); ++i)
    {
      labels.push_back(atof(tokens.at(i).c_str()));
    }
    
    lines.push_back(std::make_pair(tokens.at(0), labels));
    //###To clear the vector labels
    labels.clear();
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id].first,   //###
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

上面的代码主要有两处进行了修改：一处是读取txt文件部分， 第二处是ReadImageToDatum函数。
首先，原来的label是一个int类型的变量，现在的label是多个float类型的变量，所以就有了下面的修改：

  //std::vector<std::pair<std::string, int> > lines;  //###
  std::vector<std::pair<std::string, std::vector<float> > > lines;
  std::string line;
  //size_t pos;
  //int label;  //###
  std::vector<float> labels;
 
用float类型的vector来存放label，然后在读取txt文件的while循环中修改读取label部分的代码。
第一处修改完成之后，接下来需要对ReadImageToDatum函数进行修改，这个函数的作用是将图片的信息写入到Datum中，对Datum，Blob还不太了解的朋友可以参考下面这篇博文：http://www.cnblogs.com/yymn/articles/4479216.html，这里先暂时将Datum理解为一个存放图片信息（包括像素值和label）的数据结构，用于将图片写入到lmdb文件。
ReadImageToDatum函数在io.hpp中声明，我是使用sublime text3打开（open folder）caffe文件夹，直接选中ReadImageToDatum右键就可以“Goto Definition”。
在io.hpp文件中，原来的ReadImageToDatum函数是像下面这样声明的：

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);
 
我们可以不改动原来的函数声明（因为C++支持函数重载，这里指参数有所不同），而在它的下面接上：
bool ReadImageToDatum(const string& filename, const vector<float> labels,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);
 
容易注意到，我们参原来的参数
const int label
 修改成：
const vector<float> labels
 接着，我们需要在io.cpp函数中实现我们增加的重载函数：

bool ReadImageToDatum(const string& filename, const vector<float> labels,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    // if (encoding.size()) {
    //   if ( (cv_img.channels() == 3) == is_color && !height && !width &&
    //       matchExt(filename, encoding) )
    //     return ReadFileToDatum(filename, label, datum);
    //   std::vector<uchar> buf;
    //   cv::imencode("."+encoding, cv_img, buf);
    //   datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
    //                   buf.size()));
    //   datum->set_label(label);
    //   datum->set_encoded(true);
    //   return true;
    // }
                    
    CVMatToDatum(cv_img, datum);
    //datum->set_label(label);

    //###
    for (int i = 0; i < labels.size(); ++i)
    {
      datum->add_float_data(labels.at(i));
    }

    return true;
  } else {
    return false;
  }
}
在原来的ReadImageToDatum定义下面加上新的定义，（BTW：encoding部分对我暂时没有什么用，所以暂时注释掉）。这里使用：
datum->add_float_data(labels.at(i));
 将label写入到Datum中。
好了！经过上面的步骤，回到caffe目录下，重新make编译一下，就会在build/tools/文件夹下面生成一个convert_imageset_regression.bin可执行文件了。
再接下来制作lmdb的方法就跟分类任务一样了，需要制作我们的train.txt以及test.txt，以及将我们用于train和test的图片放到相应的文件夹下面，然后调用convert_imageset_regression.bin来制作lmdb即可，经过上面的代码修改，convert_imageset_regression.bin已经“懂得”如何将后面带有多个浮点类型的数字的txt转换成lmdb文件啦！

这里，可能有的朋友还会有一点疑问，
datum->add_float_data(labels.at(i));
 这个函数是怎么来的，第一次用的时候怎么会知道有这个函数？
这就得来看看caffe.proto文件了，里面关于Datum的代码如下：

message Datum {
  optional int32 channels = 1;
  optional int32 height = 2;
  optional int32 width = 3;
  // the actual image data, in bytes
  optional bytes data = 4;
  optional int32 label = 5;
  // Optionally, the datum could also hold float data.
  repeated float float_data = 6;
  // If true data contains an encoded image that need to be decoded
  optional bool encoded = 7 [default = false];
}

.proto文件是Google开发的一种协议接口，根据这个，可以自动生成caffe.pb.h和caffe.pb.cc文件。
其中，
  optional int32 label = 5;
 就是用于分类的。
而，
repeated float float_data = 6;
就是我们用来做回归的。
在caffe.pb.h文件中可以找到关于这部分自动生成的代码：

  // optional int32 label = 5;
  inline bool has_label() const;
  inline void clear_label();
  static const int kLabelFieldNumber = 5;
  inline ::google::protobuf::int32 label() const;
  inline void set_label(::google::protobuf::int32 value);

  // repeated float float_data = 6;
  inline int float_data_size() const;
  inline void clear_float_data();
  static const int kFloatDataFieldNumber = 6;
  inline float float_data(int index) const;
  inline void set_float_data(int index, float value);
  inline void add_float_data(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      float_data() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_float_data();

在这里就可以看到，关于操作label的一系列函数，如果我们不使用add_float_data，而是用set_float_data，也是可以的！
上篇就到这里吧。
在上篇中，我们已经实现了lmdb的制作，实际上就是将训练和测试的图片的信息存放在Datum中，然后再序列化到lmdb文件中。
上篇完成了数据的准备工作，而要跑通整个实验，还需要在data_layer.cpp中做一些相应的修改。
data_layer.cpp中的函数实现了从lmdb中读取图片信息，先是反序列化成Datum，然后再放进Blob中。仔细想一下可以知道，因为原先caffe的data_layer.cpp的实现是针对分类的情况，所以读取label部分的代码并不适用于回归的情况。
所以本篇介绍data_layer.cpp需要修改的代码，以及训练的时候需要注意的一些细节。
下面是我修改后的data_layer.cpp文件，主要修改了两处地方：一是DataLayerSetup函数，二是load_batch函数。同上篇一样，有//###标记的就是我修改的地方：

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  //###
  // if (this->output_labels_) {
  //   vector<int> label_shape(1, batch_size);
  //   top[1]->Reshape(label_shape);
  //   for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
  //     this->prefetch_[i].label_.Reshape(label_shape);
  //   }
  // }

  //###
  int labelNum = 4;
  if (this->output_labels_) {

    vector<int> label_shape;
    label_shape.push_back(batch_size);
    label_shape.push_back(labelNum);
    label_shape.push_back(1);
    label_shape.push_back(1);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    // Copy label.
    //###
    // if (this->output_labels_) {
    //   top_label[item_id] = datum.label();
    // }

    //###
    int labelNum = 4;
    if (this->output_labels_) {
      for(int i=0;i<labelNum;i++){
        top_label[item_id*labelNum+i] = datum.float_data(i); //read float labels
      }
    }


    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe

其中，第一处修改是：

  //###
  int labelNum = 4;	//标签的数量，也就是txt中每一张图后面跟着的浮点数的数目
  if (this->output_labels_) {

    vector<int> label_shape;
    label_shape.push_back(batch_size);
    label_shape.push_back(labelNum);
    label_shape.push_back(1);
    label_shape.push_back(1);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

从DataLayerSetup函数传进来的参数可以看到，top是一个向量的地址，而向量的元素是Blob<Dtype>*。因为在caffe网络结构中，图片信息是分成两个Blob进行传递的，一个Blob记录图片的像素值，另外一个Blob记录图片的标签，这里的top[0]，top[1]分别与之对应（所以实际上我们要修改的是top[1]相关的内容，top[0]相关的我们并不需要管）。
上面的代码是对top[1]的Reshape，push_back的四个值分别对应Blob的num，channels，height，width。因为top[1]对应的是标签，所以num设置为batch_size，channels设置为labelNum，height和width设置为1即可。这一步相当于是“塑造”一个适合我们数据label的Blob出来。
第二处修改的地方是：

    //###
    int labelNum = 4;
    if (this->output_labels_) {
      for(int i=0;i<labelNum;i++){
        top_label[item_id*labelNum+i] = datum.float_data(i); //read float labels
      }

这个地方是将datum中的label值赋值给top_label。
完成了上面两处修改之后，跟上篇一样，需要回到caffe目录下，重新执行make编译一下data_layer.cpp。编译完成之后，我们的修改就生效了！这样一来，convert_imageset_regression完成了将回归数据制作成lmdb的任务，而data_layer则完成了将用于回归的lmdb成功送入后续网络的任务。
那么，要成功运行caffe.bin进行训练，还需要注意一下下面的细节，主要是要注意网络配置文件（.prototxt）：
1、最后一个全连接层的num_output应该与labelNum（即label的数目相等）
2、做分类任务的时候，一般是使用SoftmaxWithLoss类型的loss层，而在做回归任务的时候，一般是用EuclideanLoss类型的loss层，因为loss主要体现在网络最后一个全连接层的输出与ground true的欧氏距离
3、不使用Accuracy层，因为回归任务没有所谓的准确率
4、如果要在数据层做crop，scale，mirror等操作，应该先考虑一下变换之后你的label是否也需要变化，不能像分类任务那么“直接”地用
5、修改data_layer.cpp并重新编译之后，下次如果要进行分类任务，得记得改回去并重新编译（或者可以在github上git clone多个caffe下来，这样就不用来回修改）。

完成了上面所有的工作之后就可以对自己的数据进行训练和测试了。训练之后得到caffemodel，就可以拿来应用了。应用的时候，可以用caffe的Python接口或者是继续修改源码。
在这里，要感谢超哥的指点，使我看caffe代码的时候容易了许多。下面是他的博客以及github：
his CSDN
his github

//###
发现了一篇用caffe做多标签分类的博文，改代码的思路很相似，可以互相借鉴：
http://blog.csdn.net/hubin232/article/details/50960201
