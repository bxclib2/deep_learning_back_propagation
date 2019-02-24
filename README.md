This is the code of EPFL master course Deep learning

现在有两个branch

master branch 是一开始用循环写的，batch_compute是直接用矩阵写的（我后加的）。

经测试，效率差距非常大

master 循环版本：

softmax+交叉熵：需要69.943792s 完成1000次GD

sigmoid+MSE:需要65.353737s 完成1000次GD

batch_compute 矩阵版本：

softmax+交叉熵：需要9.465829s 完成1000次GD

sigmoid+MSE:仅仅需要2.6018209999999993s 完成1000次GD

之所以第二个版本的softmax 网络比较慢的原因是，softmax层的反向传播我没有想到不用循环就能解决的方式。如果能去掉那个循环，我觉得应该也就2s左右就算完了。第一个全是循环的版本每个大概都要1分钟。

那个循环的位置，大家头脑风暴一下，看看能不能解决掉。

如果想阅读代码的话，第一个版本会easy一点。最后上交应该会交第二个，因为速度实在差太多了。

循环已经被我想办法去掉了,目前软回归的速度大概是MSE 的一半了,都非常迅速.