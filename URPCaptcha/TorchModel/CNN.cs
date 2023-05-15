using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils.data;

namespace URPCaptcha.TorchModel
{
    public class CNN
    {
        private static int _epochs = 10;
        private static int _batchSize = 64;
        private readonly static int _logInterval = 100;

        public static void Run(string[] args)
        {
            var trainset = args[0];
            var testset = args[1];
            var trainsetPath = Path.GetFullPath(trainset);
            var testsetPath = Path.GetFullPath(testset);

            random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = cuda.is_available() ? CUDA : CPU;
            Console.WriteLine($"Running CNN on {device.type}");

            if (device.type == DeviceType.CUDA)
            {
                _batchSize *= 4;
            }

            using var model = new Model("model", device);

            var normImage = torchvision.transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 });

            using (MyDataset train_data = new MyDataset(trainsetPath),
                           test_data = new MyDataset(testsetPath))
            {
                TrainingLoop("cnn", device, model, train_data, test_data);
            }

            model.save(args[2]);
        }

        internal static void TrainingLoop(string dataset, Device device, Model model, MyDataset train_data, MyDataset test_data)
        {
            using var train = new DataLoader(train_data, _batchSize, device: device, shuffle: true);
            using var test = new DataLoader(test_data, _batchSize, device: device, shuffle: false);

            if (device.type == DeviceType.CUDA)
            {
                _epochs *= 4;
            }

            var optimizer = optim.Adam(model.parameters());

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++)
            {

                using (var d = NewDisposeScope())
                {

                    Train(model, optimizer, MultiLabelSoftMarginLoss(), train, epoch, train_data.Count);
                    Test(model, MultiLabelSoftMarginLoss(), test, test_data.Count);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                    optimizer.step();
                }
            }

            sw.Stop();
            Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s.");

            Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
            model.save(dataset + ".model.bin");
        }
        private static void Train(
            Model model,
            optim.Optimizer optimizer,
            Loss<Tensor, Tensor, Tensor> loss,
            DataLoader dataLoader,
            int epoch,
            long size)
        {
            model.train();

            int batchId = 1;
            long total = 0;

            Console.WriteLine($"Epoch: {epoch}...");

            using (var d = torch.NewDisposeScope())
            {

                foreach (var data in dataLoader)
                {
                    optimizer.zero_grad();

                    var target = data["label"];
                    var prediction = model.call(data["image"]);
                    var output = loss.call(prediction, target);

                    output.backward();

                    optimizer.step();

                    total += target.shape[0];

                    if (batchId % _logInterval == 0 || total == size)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;

                    d.DisposeEverything();
                }
            }
        }
        private static void Test(
            Model model,
            Loss<Tensor, Tensor, Tensor> loss,
            DataLoader dataLoader,
            long size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            using (var d = NewDisposeScope())
            {

                foreach (var data in dataLoader)
                {
                    var prediction = model.call(data["image"]);
                    var output = loss.call(prediction, data["label"]);
                    testLoss += output.ToSingle();

                    correct += Model.GetAcc(prediction, data["label"]);
                    //correct += pred.eq(data["label"].argmax(1)).sum().ToInt32();

                    d.DisposeEverything();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }
    }
    public class Model : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> net;
        private readonly Module<Tensor, Tensor> fc;

        public Model(string name, Device device = null) : base(name)
        {

            net = Sequential(
                // 64*3*180*60
                Conv2d(3, 16, kernelSize: 3, padding: 1),
                MaxPool2d(kernelSize: 2, stride: 2),
                BatchNorm2d(16),
                ReLU(),
                // batch_size * channels * width * height = 64*16*50*19
                // 64*16*90*30
                Conv2d(16, 64, kernelSize: 3, padding: 1),
                MaxPool2d(kernelSize: 2, stride: 2),
                BatchNorm2d(64),
                ReLU(),
                // batch_size * channels * width * height = 64*64*25*9
                // 64*64*45*15

                Conv2d(64, 256, kernelSize: 3, padding: 1),
                MaxPool2d(kernelSize: 2, stride: 2),
                BatchNorm2d(256),
                ReLU(),
                // 64*256*22*7

                Conv2d(256, 512, kernelSize: 3, padding: 1),
                MaxPool2d(kernelSize: 2, stride: 2),
                BatchNorm2d(512),
                ReLU()
            // batch_size * channels * width * height = 64*512*12*4
            // 64*512*11*3
            );

            fc = Sequential(Linear(512 * 11 * 3, 36 * 4));

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor input)
        {
            return fc.forward(
                net.forward(input)
                    .view(-1, 512 * 11 * 3)
            );
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                net.Dispose();
                fc.Dispose();
            }
            base.Dispose(disposing);
        }

        public static int GetAcc(Tensor predict, Tensor label)
        {
            predict = predict.view(-1, 36);
            label = label.view(-1, 36);

            var p = predict.argmax(1).view(-1, 4);
            var l = label.argmax(1).view(-1, 4);

            int result = 0;

            for (int i = 0; i < p.shape[0]; i++)
            {
                if (equal(p[i], l[i]).sum().ToInt32() == 4)
                {
                    result++;
                }
            }

            return result;

            //return p.equal(l).sum().ToInt32();
        }
    }
}
