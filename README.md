# URPCaptcha

URP教务处验证码自动填写, [配套油猴脚本](https://greasyfork.org/zh-CN/scripts/466346)

使用[TorchSharp](https://github.com/dotnet/TorchSharp)和[ASP.NET Core](https://github.com/dotnet/aspnetcore)重写
[HEBUT_Administration_Captcha](https://github.com/WangJerry1229/HEBUT_Administration_Captcha).

感谢[匿名用户](https://github.com/Gerchart-GXT)提供的服务器用于部署.

## 模型训练

使用以下语句

```C#
CNN.Run(new string[]{trainDatasetPath, testDatasetPath, modelSavePath });
```

需要使用CUDA进行训练或部署需要将Nuget中的[TorchSharp-cpu](https://www.nuget.org/packages/TorchSharp-cpu)卸载, 
安装[TorchSharp-cuda-windows](https://www.nuget.org/packages/TorchSharp-cuda-windows)
或[TorchSharp-cuda-linux](https://www.nuget.org/packages/TorchSharp-cuda-linux). 

## 部署建议

运行时较大(使用CPU在`200MiB`左右, 使用CUDA在`2.5GiB`左右, Linux翻倍), 若对部署包有体积限制请合理选择. 

内存占用约`100MiB`

**仅支持x64!!! 仅支持x64!!! 仅支持x64!!!**(点名批评Azure Web App Services免费或共享层仅提供32位环境)
