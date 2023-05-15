using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using URPCaptcha.TorchModel;
using static TorchSharp.torchvision.io;

namespace URPCaptcha
{
    public class Program
    {
        // Solve CORS Problem
        [HttpOptions]
        private static IResult CORS(HttpContext context)
        {
            /*context.Response.Headers.Add("Access-Control-Allow-Origin", new[] { "*" });
            context.Response.Headers.Add("Access-Control-Allow-Headers", new[] { "*" });
            context.Response.Headers.Add("Access-Control-Allow-Methods", new[] { "GET, POST, PUT, DELETE, OPTIONS" });
            context.Response.Headers.Add("Access-Control-Allow-Credentials", new[] { "true" });*/

            return Results.Ok();
        }
        public static void Main(string[] args)
        {
            if (!Directory.Exists("./reports"))
            {
                Directory.CreateDirectory("./reports");
            }

            Model model = new Model("model");
            model.load("model.dat");
            model.eval();

            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddAuthorization();

            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            app.UseRouting();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            #region CROS
            app.Use(async (context, next) =>
            {
                context.Response.Headers.Add("Access-Control-Allow-Origin", new[] { "*" });
                context.Response.Headers.Add("Access-Control-Allow-Headers", new[] { "*" });
                context.Response.Headers.Add("Access-Control-Allow-Methods", new[] { "GET, POST, PUT, DELETE, OPTIONS" });
                context.Response.Headers.Add("Access-Control-Allow-Credentials", new[] { "true" });

                await next.Invoke();
            });
            #endregion

            app.UseAuthorization();

            #region predict
            app.Map("/predictfile", CORS).WithOpenApi();
            app.MapPost("/predictfile", (IFormFile file) =>
            {
                if (file.ContentType is not "image/png")
                {
                    return Results.StatusCode(415);
                }
                var image = read_image(file.OpenReadStream(), ImageReadMode.RGB, new SkiaImager(100)).@float().view(1, 3, 60, 180); 
                var result = MyDataset.myDecode(model.call(image));
                return Results.Ok(result);
            })
            .WithName("PredictFile")
            .WithOpenApi();

            app.Map("/predictbase64", CORS).WithOpenApi();
            // use "application/json" to pass parameter
            // example(only quotations and string): "base64"
            app.MapPost("/predictbase64", ([FromBody]string base64) =>
            {
                var bytes = Convert.FromBase64String(base64);
                var image = read_image(new MemoryStream(bytes), ImageReadMode.RGB, new SkiaImager(100)).@float().view(1, 3, 60, 180);
                var result = MyDataset.myDecode(model.call(image));
                return Results.Ok(result);
            })
            .WithName("PredictBase64")
            .WithOpenApi();
            #endregion

            #region ReportError
            app.Map("/reporterrorfile", CORS).WithOpenApi();
            app.MapPost("/reporterrorfile", async (IFormFile file) =>
            {
                if (file.ContentType is not "image/png")
                {
                    return Results.StatusCode(415);
                }
                using (var image = file.OpenReadStream())
                {
                    string filename = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
                    byte[] data = new byte[image.Length];
                    await image.ReadAsync(data, 0, data.Length);
                    using (var fs = new FileStream($"./reports/{filename}.png", FileMode.OpenOrCreate))
                    {
                        await fs.WriteAsync(data, 0, data.Length);
                    }
                }
                return Results.Ok();
            })
            .WithName("ReportErrorFile")
            .WithOpenApi();
            
            app.Map("/reporterrorbase64", CORS).WithOpenApi();
            app.MapPost("/reporterrorbase64", async ([FromBody] string base64) =>
            {
                string filename = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
                byte[] data = Convert.FromBase64String(base64);
                using (var fs = new FileStream($"./reports/{filename}.png", FileMode.OpenOrCreate))
                {
                    await fs.WriteAsync(data, 0, data.Length);
                }
                return Results.Ok();
            })
            .WithName("ReportErrorBase64")
            .WithOpenApi();

            app.Map("/reportscount", CORS).WithOpenApi();
            app.MapPost("/reportscount", () =>
            {
                var dir = new DirectoryInfo("./reports");
                long count = dir.GetFiles().LongLength;
                long length = dir.EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
                return Results.Ok($"totally {count} files with {length} bytes");
            })
            .WithName("ReportsCount")
            .WithOpenApi();

            app.Map("/cleanreports", CORS).WithOpenApi();
            app.MapPost("/cleanreports", () =>
            {
                var files = new DirectoryInfo("./reports").GetFiles();
                foreach (var file in files)
                {
                    file.Delete();
                }
                return Results.Ok();
            })
            .WithName("CleanReports")
            .WithOpenApi();
            #endregion

            app.Run();
        }
    }
}