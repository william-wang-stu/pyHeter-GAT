### Gitee 仓库提交
主要记录下使用Gitee仓库管理源代码的笔记
1. 在当前机器下(这里是10mul3090远程服务器上)使用`ssh-keygen`命令生成ssh公钥, 完整命令为`ssh-keygen -t rsa -C wangzejian_1215@163.com`, 其中最后的邮箱是我gitee账号的绑定邮箱(如果不主动绑定的话gitee提示说有个缺省邮箱可以使用, 但我没试过...)
2. 将生成的公钥(文件路径为~/.ssh/id_rsa.pub)中的内容复制到gitee账户的[SSH公钥管理](https://gitee.com/profile/sshkeys)下
3. 在当前机器下使用`ssh -T git@gitee.com`命令验证结果, 正常结果应类似`Hi wangzejian1120! You've successfully authenticated, but GITEE.COM does not provide shell access.`
4. 在当前机器的对应仓库文件夹下关联gitee远程仓库`git remote add gitee git@gitee.com:wangzejian1120/py-heter-gat.git`(不清楚是不是一定要用ssh协议访问, 没试过http行不行...), 并按需执行git指令即可

### Github 仓库提交
显然, 1)在Git账户下添加当前机器生成的SSH公钥, 2)使用ssh协议(.git后缀)访问远程仓库, 能够克服Github仓库网络连接的问题
同时, 经过测试, 仅使用命令`ssh-keygen`生成的公钥也是能行的, 不用一定加上`-C 邮箱`的
