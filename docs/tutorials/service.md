# Service Tutorial
This tutorial describes how to quickly serve a NER model by using docker.

## Build
This section is for thoses who don't have a docker image named `carina/kaner`, which provides a short command to build a docker image from source code.

```bash
sudo docker build -t carina/kaner ./
```

## Run
If you already have a docker image named `carina/kaner`, you can run it quickly by the following command.

```bash
docker run -it -p 8080:8080 carina/kaner --mfolder ./data/logs/blcrf-ccksfeee --host 0.0.0.0 --port 8080
```

It is important to note that we can also pass several arguments to change the configuration of this service, such as *host*, *port*, and *model_folder*. More details can be found in the following table.

<table>
  <tr>
    <th>Argument</th>
    <th>Values</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>mfolder</td>
    <td>./data/logs/blcrf-<a href="https://www.biendata.xyz/competition/ccks_2020_4_2/data/">ccksfeee</a> | ./data/logs/ses-<a href="https://www.biendata.xyz/competition/ccks_2020_4_2/data/">ccksfeee</a></td>
    <td>It means the folder where the trained model exists. Different folders may representate different models on different datasets. For example, <b>blcrf+ccksfeee</b> means a BiLSTM-CRF based model trained on the dataset <b>ccksfeee</b>.</td>
  </tr>
  <tr>
    <td>host</td>
    <td>0.0.0.0 | 127.0.0.1</td>
    <td>On the Internet, a host address is the IP address of the machine where the service serves. </td>
  </tr>
  <tr>
    <td>port</td>
    <td>1024 - 49151</td>
    <td>The service will listen on this port.</td>
  </tr>
</table>

If you see the following message, then you have succeefully started the service!

```plain
->    :: Web Service ::
service starts at http://0.0.0.0:8080
```

## Test
Now, you can use a **POST** tool like [`Hoppscotch`](https://hoppscotch.io/) or `Linux Curl` to test the model. Our service accepts data with json format that has a key named **texts**. The value of the key **texts** is a list of text that you want to predict. Additionally, we do not support parsing long document with length greater than *512* so far. Hence, you have to limit the length of the text in the list. Enjoy the tool!

```bash
curl -X POST -H "Content-Type: application/json" http://0.0.0.0:8080/kaner/predict -d '{"texts": ["红日药业：关于持股5% 以上的股东增持公司股份比例达到4%的公告<br>天津红日药业股份有限公司<br>证券代码：300026 证券简称：红日药业 公告编号：2019-085天津红日药业股份有限公司<br>关于持股5%以上的股东增持公司份比例达到 4%的公告<br>本公司及董事会全体成员保证信息披露内容的真实、准确和完整，没有虚假记载、误导性陈述或重大遗漏。<br>天津红日药 业股份有限公司（以下简称“公司”）于2019年9月19日收到公<br>司持股5%以上的股东成都兴城投资集团有限公司（以下简称“兴城集团”）出具<br>的关于增持公司股份的通知， 兴城集团自2019年5月6日至2019年9月19日期间，通过深圳证券交易所系统累计增持公司股份121374322股，增持比例达到公司总股本的4%，现将相关情况公告如下：<br>一、本次股东增持股份情况<br>1、增持人：成都兴城投资集团有限公司<br>2、增持目的：基于对公司未来发展充满信心，对公司价值的高度认可，促<br>进公司健康可持续发展，提升上市公司投资价值。<br>3、增持方式：通过深圳证券交易所系统以集中竞价交易方式增持。"]}'
```


If all steps are well done, you can see the following outputs.

```json
{
    "success": true,
    "data": [{
        "text": "红日药业：关于持股5% 以上的股东增持公司股份比例达到4%的公告<br>天津红日药业股份有限公司<br>证券代码：300026 证券简称：红日药业 公告编号：2019-085天津红日药业股份有限公司<br>关于持股5%以上的股东增持公司份比例达到 4%的公告<br>本公司及董事会全体成员保证信息披露内 容的真实、准确和完整，没有虚假记载、误导性陈述或重大遗漏。<br>天津红日药 业股份有限公司（以下简称“公司”）于2019年9月19日收到公<br>司持股5%以上的股东成都兴城投资集团有限公司（以下简称“兴城集团”）出具<br>的关于增持公司股份的通知， 兴城集团自2019年5月6日至2019年9月19日期间，通过深圳证券交易所系统累计增持公司股份121374322股，增持比例达到公司总股本的4%，现将相关情况公告如下：<br>一、本次股东增持股 份情况<br>1、增持人：成都兴城投资集团有限公司<br>2、增持目的：基于对公司未来发展充满信心，对公司价值的高度认可，促<br>进公司健康可持续发展，提升上市公司投资价值。<br>3、增持方式：通过深圳证券交易所系统以集中竞价交易方式增持。",
        "spans": [{
                "start": 236,
                "end": 247,
                "label": "股东增持\t增持的股东",
                "text": "成都兴城投资集团有限公司",
                "confidence": 1.0
            },
            {
                "start": 391,
                "end": 402,
                "label": "股东增持\t增持的股东",
                "text": "成都兴城投资集团有限公司",
                "confidence": 1.0
            }
        ]
    }],
    "model": "blcrf",
    "dataset": "ccksfeee",
    "time": 0
}
```