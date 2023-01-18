using Common;
using DomainModel;
using System.Diagnostics;
using System.Globalization;

namespace ArtificialIntelligenceTools
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
        public static double GetSubquestionTemplateSuggestedPoints(string login, bool retrainModel, SubquestionTemplateRecord subquestionTemplateRecord, EnumTypes.Model usedModel)
        {
            string function = "predict_new";
            string[] arguments = new string[] { subquestionTemplateRecord.SubquestionTypeAveragePoints.ToString(), subquestionTemplateRecord.CorrectAnswersShare.ToString(),
            subquestionTemplateRecord.SubjectAveragePoints.ToString(), subquestionTemplateRecord.ContainsImage.ToString(), subquestionTemplateRecord.NegativePoints.ToString(), subquestionTemplateRecord.MinimumPointsShare.ToString()};
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
            fileName = "TemplateNeuralNetwork.py";
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "ArtificialIntelligenceTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
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
                    try
                    {
                        //in certain rare cases, the amount of suggested subquestion points could be less than 0
                        double parsedResult = Math.Round(Convert.ToDouble(result, CultureInfo.InvariantCulture), 2);
                        if(parsedResult < 0)
                        {
                            parsedResult = 2;
                        }
                        return parsedResult;
                    }
                    catch
                    {
                        throw Exceptions.PythonException(fileName, function, stderr);
                    }
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
        public static double GetSubquestionResultSuggestedPoints(string login, bool retrainModel, SubquestionResultRecord subquestionResultRecord, EnumTypes.Model usedModel)
        {
            string function = "predict_new";
            string[] arguments = new string[] { subquestionResultRecord.SubquestionTypeAveragePoints.ToString(), subquestionResultRecord.AnswerCorrectness.ToString(),
            subquestionResultRecord.SubjectAveragePoints.ToString(), subquestionResultRecord.ContainsImage.ToString(), subquestionResultRecord.NegativePoints.ToString(), subquestionResultRecord.MinimumPointsShare.ToString()};
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
            fileName = "ResultNeuralNetwork.py";
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "ArtificialIntelligenceTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
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
                    try
                    {
                        double parsedResult = Math.Round(Convert.ToDouble(result, CultureInfo.InvariantCulture), 2);
                        //in certain rare cases, the amount of suggested student's points could be more than the maximum possible value (or less than minimum)
                        if(parsedResult > subquestionResultRecord.SubquestionResult.SubquestionTemplate.SubquestionPoints)
                        {
                            parsedResult = subquestionResultRecord.SubquestionResult.SubquestionTemplate.SubquestionPoints;
                        }
                        else if (parsedResult < subquestionResultRecord.SubquestionResult.SubquestionTemplate.SubquestionPoints * (-1))
                        {
                            parsedResult = subquestionResultRecord.SubquestionResult.SubquestionTemplate.SubquestionPoints * (-1);
                        }
                        return parsedResult;
                    }
                    catch
                    {
                        throw Exceptions.PythonException(fileName, function, stderr);
                    }
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
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "ArtificialIntelligenceTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, function);
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
                    try
                    {
                        result = result.Substring(0, result.Length - 4);//remove new line from the result
                        return Math.Round(Convert.ToDouble(result, CultureInfo.InvariantCulture), 4);
                    }
                    catch
                    {
                        throw Exceptions.PythonException(fileName, function, stderr);
                    }
                }
            }
        }

        /// <summary>
        /// Predicts the amount of points that the average student will get for the test
        /// <param name="retrainModel">Indicates whether the model should be retrained or not</param>
        /// <param name="login">Login of the user</param>
        /// <param name="usedModel">Model to be used</param>
        /// <param name="testTemplateId">Identifier of the test</param>
        /// </summary>
        public static double GetTestTemplatePredictedPoints(bool retrainModel, string login, EnumTypes.Model usedModel, int testTemplateId)
        {
            string fileName;
            if(usedModel == EnumTypes.Model.MachineLearning)
            {
                fileName = "DifficultyMachineLearning.py";
            }
            else
            {
                fileName = "DifficultyNeuralNetwork.py";
            }
            fileName = "DifficultyNeuralNetwork.py";
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4}",
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "ArtificialIntelligenceTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + fileName + " ", Config.SelectedPlatform.ToString(), login, retrainModel, testTemplateId);
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
                    try
                    {
                        return double.Parse(result, CultureInfo.InvariantCulture);
                    }
                    catch
                    {
                        throw Exceptions.PythonException(fileName, "N/A", stderr);
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
                Config.GetPythonScriptsPath() + Config.GetPathSeparator() + "ArtificialIntelligenceTools" + Config.GetPathSeparator() + "PythonScripts" + Config.GetPathSeparator() + "OtherFunctions.py ", function);
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

                    if(stderr.Length > 0)
                    {
                        throw Exceptions.PythonException("OtherFunctions.py", function, stderr);
                    }

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