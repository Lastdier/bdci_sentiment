# bdci_sentiment

## 数据集基本情况
    
* 文件基本说明
    * 初赛共有 9947 个训练样本，2365 个测试样本
    * 复赛共有 12572 个训练样本，2753 个测试样本
    * 第0列 (content_id) ：文本 id  
    * 第1列 (article)    ：「字」级别上的表示 
    * 第2列 (word_seg)   ：「词」级别上的表示  
    
       
* 类别统计
    * 主题统计（共 10 个主题）
      * 初赛训练集
        * 动力: 2732, 价格: 1273, 油耗: 1082, 操控: 1036, 舒适性: 931
        * 配置: 853, 安全性: 573, 内饰: 536, 外观: 489, 空间: 442
      * 复赛训练集
        * 动力: 3454, 价格: 1634, 油耗: 1379, 操控: 1302, 舒适性: 1182
        * 配置: 1075, 安全性: 736, 内饰: 669, 外观: 606, 空间: 535
    
    * 情感统计（情感分 3 类）
      * 初赛训练集：-1: 1616, 0: 6661, 1: 1670
      * 初赛测试集：-1: 421,  0: 1824, 1: 380
      * 复赛训练集：-1: 2036, 0: 8488, 1: 2048
      
    * 主题数量统计
      * 初赛测试集：1: 2146, 2: 180, 3: 33, 4: 5, sum: 2625
      * 复赛测试集：1: 2572, 2: 154, 3: 20, 4: 6, 5: 1, sum: 2969
      * 复赛训练集：1: 9182, 2: 1130, 3: 264, 4: 57, 5: 17, 6: 3, 7: 1, sum: 12572
      

## Pycharm 建议设置

* 换行符统一为'\n'
  * File | Settings | Editor | Code Style
    * `Line separator` 选择 `Unix and OS X (\n)`

* 编码统一为`UTF-8`
  * File | Settings | Editor | File Encodings
    * Global 和 Project 都选择 `UTF-8`
    * 下方 Default encoding for properties files也选择 `UTF-8`

* Python Doc 使用 Google 风格
  * File | Settings | Tools | Python Integrated Tools
    * `Docstring format` 选择 `Google`

* 设置Python 默认代码
  * File | Settings | Editor | File and Code Templates
    * 选择 `Python Script` 粘贴以下内容, `${USER}`可换为自己想显示的昵称
    * 可以自己按需修改
    
## 其他
use ELMo pretrain from HIT-SCIR
https://github.com/HIT-SCIR/ELMoForManyLangs
