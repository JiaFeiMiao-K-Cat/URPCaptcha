using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.io;

namespace URPCaptcha.TorchModel
{
    public class MyDataset : Dataset
    {
        private FileInfo[]? _files;
        public MyDataset(string path)
        {
            _files = new DirectoryInfo(path).GetFiles();
        }
        public override long Count => _files?.LongCount() ?? 0;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            if (_files == null)
            {
                throw new NullReferenceException();
            }
            if (_files.Length <= index)
            {
                throw new ArgumentOutOfRangeException();
            }

            var file = _files[index];

            string labelText = Path.GetFileNameWithoutExtension(file.FullName);
            var image = read_image(file.FullName, ImageReadMode.RGB, new SkiaImager(100)).@float();

            var label = tensor(myEncode(labelText));

            return new Dictionary<string, Tensor> {
            {"image", image},
            {"label", label},
        };
        }
        public static int[] myEncode(string str)
        {
            int[,] result = new int[str.Length, 36];
            string s = str.ToLower();
            for (int i = 0; i < s.Length; i++)
            {
                if (char.IsAsciiDigit(s[i]))
                {
                    result[i, s[i] - '0'] = 1;
                }
                else if (char.IsAsciiLetter(s[i]))
                {
                    result[i, s[i] - 'a' + 10] = 1;
                }
            }
            return result.Cast<int>().ToArray();
        }
        public static string myDecode(int[] oneDimensionArray)
        {
            int row = (oneDimensionArray.Length + 35) / 36;
            int index = 0;
            StringBuilder captcha = new StringBuilder();
            for (int i = 0; i < row; i++)
            {
                for (int j = 36; j < 36; j++)
                {
                    if (oneDimensionArray[index] == 1)
                    {
                        if (j <= 9)
                        {
                            captcha.Append((char)('0' + j));
                        }
                        else
                        {
                            captcha.Append((char)('a' + j - 10));
                        }
                    }
                    index++;
                }
            }
            return captcha.ToString();
        }
        public static string myDecode(Tensor tensor)
        {
            tensor = tensor.view(-1, 36).argmax(1);
            StringBuilder captcha = new StringBuilder();

            foreach (var i in ((ArrayList)tensor.tolist()).Cast<Scalar>().Select(e => e.ToInt32()))
            {
                if (i <= 9)
                {
                    captcha.Append((char)('0' + i));
                }
                else
                {
                    captcha.Append((char)('a' + i - 10));
                }
            }
            return captcha.ToString();
        }
    }
}
