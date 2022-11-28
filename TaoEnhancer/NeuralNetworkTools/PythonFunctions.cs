using Common;
using DomainModel;
using System.Diagnostics;

namespace NeuralNetworkTools
{
    public class PythonFunctions
    {
        /// <summary>
        /// Suggests the appropriate amount of points that the subquestion template should have
        /// <param name="login">Login of the user</param>
        /// <param name="retrainModel">Indicates whether the model should be retrained or not</param>
        /// <param name="subquestionTemplateRecord">Subquestion template whose points we want to predict (converted to subquestionTemplateRecord form)</param>
        /// <param name="usedModel">Model used to make the prediction</param>
        /// </summary>
        public static string GetSubquestionTemplateSuggestedPoints(string login, bool retrainModel, SubquestionTemplateRecord subquestionTemplateRecord, EnumTypes.Model usedModel)
        {
            string function = "predict_new";
            string[] arguments = new string[] { subquestionTemplateRecord.SubquestionTypeAveragePoints.ToString().Replace(",", "."), subquestionTemplateRecord.CorrectAnswersShare.ToString().Replace(",", "."),
            subquestionTemplateRecord.SubjectAveragePoints.ToString().Replace(",", "."), subquestionTemplateRecord.ContainsImage.ToString().Replace(",", "."), subquestionTemplateRecord.NegativePoints.ToString().Replace(",", "."), subquestionTemplateRecord.MinimumPointsShare.ToString().Replace(",", ".")};
            string fileName;
            if(usedModel == EnumTypes.Model.MachineLearning)
            {
                fileName = "TemplateMachineLearning.py";
            }
            else
            {
                //neural network is used by default
                fileName = "TemplateNeuralNetwork.py";
            }

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "NeuralNetworkTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd();
                    string result = reader.ReadToEnd();
                    return result;
                }
            }
        }

        /// <summary>
        /// Suggests the appropriate amount of points that the subquestion template should have
        /// <param name="login">Login of the user</param>
        /// <param name="retrainModel">Indicates whether the model should be retrained or not</param>
        /// <param name="subquestionResultRecord">Subquestion result whose points we want to predict (converted to subquestionResultRecord form)</param>
        /// <param name="usedModel">Model used to make the prediction</param>
        /// </summary>
        public static string GetSubquestionResultSuggestedPoints(string login, bool retrainModel, SubquestionResultRecord subquestionResultRecord, EnumTypes.Model usedModel)
        {
            string function = "predict_new";
            string[] arguments = new string[] { subquestionResultRecord.SubquestionTypeAveragePoints.ToString().Replace(",", "."), subquestionResultRecord.AnswerCorrectness.ToString().Replace(",", "."),
            subquestionResultRecord.SubjectAveragePoints.ToString().Replace(",", "."), subquestionResultRecord.ContainsImage.ToString().Replace(",", "."), subquestionResultRecord.NegativePoints.ToString().Replace(",", "."), subquestionResultRecord.MinimumPointsShare.ToString().Replace(",", ".")};
            string fileName;
            if (usedModel == EnumTypes.Model.MachineLearning)
            {
                fileName = "ResultMachineLearning.py";
            }
            else
            {
                //neural network is used by default
                fileName = "ResultNeuralNetwork.py";
            }

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "NeuralNetworkTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd();
                    string result = reader.ReadToEnd();
                    return result;
                }
            }
        }

        /// <summary>
        /// Returns the accuracy (R-squared score) of the neural network
        /// <param name="retrainModel">Indicates whether the model should be retrained or not</param>
        /// <param name="login">Login of the user</param>
        /// <param name="fileName">File containing the model to be tested</param>
        /// </summary>
        public static double GetNeuralNetworkAccuracy(bool retrainModel, string login, string fileName)
        {
            string function = "get_accuracy";
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "NeuralNetworkTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd();
                    string result = reader.ReadToEnd();//TODO: throw exception pokud stderr.Length > 0
                    try
                    {
                        result = result.Substring(0, result.Length - 4);//remove new line from the result
                        if(Config.SelectedPlatform == EnumTypes.Platform.Windows)
                        {
                            result = result.Replace(".", ",");
                        }
                        return Math.Round(Convert.ToDouble(result), 4);
                    }
                    catch
                    {
                        //todo: throw exception - chyba pri ziskavani presnosti
                        return 0;
                    }
                }
            }
        }

        public static string GetDevice()
        {
            string function = "device_name";
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "NeuralNetworkTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + "OtherFunctions.py ", function);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd();
                    string result = reader.ReadToEnd();

                    if (Config.SelectedPlatform == EnumTypes.Platform.Windows)
                    {
                        result = result.Substring(0, result.Length - 2);//remove new line from the result
                    }
                    return result;
                }
            }
        }
    }
}