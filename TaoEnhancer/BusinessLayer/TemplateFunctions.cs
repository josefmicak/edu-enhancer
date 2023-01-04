using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using ArtificialIntelligenceTools;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Xml;
using static Common.EnumTypes;
using Microsoft.AspNetCore.Http;
using Microsoft.VisualBasic;
using static System.Net.Mime.MediaTypeNames;
using System.Xml.Linq;

namespace BusinessLayer
{
    /// <summary>
    /// Functions related to test templates (entities: TestTemplate, QuestionTemplate, SubquestionTemplate)
    /// </summary>
    public class TemplateFunctions
    {
        private DataFunctions dataFunctions;

        public TemplateFunctions(CourseContext context)
        {
            dataFunctions = new DataFunctions(context);
        }

        public DbSet<TestTemplate> GetTestTemplateDbSet()
        {
            return dataFunctions.GetTestTemplateDbSet();
        }

        public List<TestTemplate> GetTestTemplatesByLogin(string login)
        {
            return dataFunctions.GetTestTemplatesByLogin(login);
        }

        public IQueryable<TestTemplate> GetTestTemplates(string login)
        {
            return GetTestTemplateDbSet()
                .Where(t => t.OwnerLogin == login);
        }

        public DbSet<QuestionTemplate> GetQuestionTemplateDbSet()
        {
            return dataFunctions.GetQuestionTemplateDbSet();
        }

        public DbSet<SubquestionTemplate> GetSubquestionTemplateDbSet()
        {
            return dataFunctions.GetSubquestionTemplateDbSet();
        }

        public DbSet<SubquestionTemplateStatistics> GetSubquestionTemplateStatisticsDbSet()
        {
            return dataFunctions.GetSubquestionTemplateStatisticsDbSet();
        }

        public DbSet<Subject> GetSubjectDbSet()
        {
            return dataFunctions.GetSubjectDbSet();
        }

        public IQueryable<Subject> GetSubjects()
        {
            return GetSubjectDbSet();
        }

        public Subject? GetSubjectById(int subjectId)
        {
            return GetSubjectDbSet().Include(s => s.StudentList).FirstOrDefault(s => s.Id == subjectId);
        }

        public async Task<string> AddSubject(Subject subject)
        {
            return await dataFunctions.AddSubject(subject);
        }

        public async Task<string> EditSubject(Subject subject, User user)
        {
            Subject? oldSubject = GetSubjectById(subject.Id);
            if(oldSubject == null)
            {
                return "Chyba: předmět nebyl nalezen.";
            }
            else if(oldSubject.GuarantorLogin != user.Login && user.Role != Role.MainAdmin)
            {
                return "Chyba: na tuto akci nemáte oprávnění";
            }
            else if (ValidateSubject(subject) != null)
            {
                return ValidateSubject(subject)!;
            }
            else
            {
                try
                {
                    oldSubject.Name = subject.Name;
                    oldSubject.Abbreviation = subject.Abbreviation;
                    oldSubject.StudentList = subject.StudentList;

                    await dataFunctions.SaveChangesAsync();
                    return "Předmět byl úspěšně upraven.";
                }
                catch(Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                    return "Při úpravě předmětu došlo k nečekané chybě.";
                }
            }
        }

        public string? ValidateSubject(Subject subject)
        {
            string? errorMessage = null;
            if(subject.Name == null || subject.Abbreviation == null)
            {
                errorMessage = "Chyba: U předmětu schází určité údaje.";
            }

            return errorMessage;
        }

        public async Task<string> DeleteSubject(Subject? subject, User user)
        {
            if(subject != null)
            {
                if (subject.GuarantorLogin != user.Login && user.Role != Role.MainAdmin)
                {
                    return "Chyba: na tuto akci nemáte oprávnění.";
                }
                else
                {
                    return await dataFunctions.DeleteSubject(subject);
                }
            }
            else
            {
                return "Chyba: neplatná operace.";
            }
        }

        public async Task<string> AddTestTemplates(string login)
        {
            List<TestTemplate> testTemplates = LoadTestTemplates(login);

            return await dataFunctions.AddTestTemplates(testTemplates, testTemplates[0].Owner);
        }

        public async Task<string> AddTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            if(subjectId == "")
            {
                return "Chyba: nevyplněný předmět.";
            }
            else
            {
                Subject? subject = GetSubjectById(int.Parse(subjectId));
                if(subject == null)
                {
                    return "Chyba: předmět nenalezen.";
                }
                else
                {
                    testTemplate.TestNameIdentifier = "tna";
                    testTemplate.TestNumberIdentifier = "tnu";
                    testTemplate.QuestionTemplateList = new List<QuestionTemplate>();
                    testTemplate.Subject = subject;

                    return await dataFunctions.AddTestTemplate(testTemplate);
                }
            }
        }

        public async Task<string> DeleteTestTemplates(string login)
        {
            return await dataFunctions.DeleteTestTemplates(login);
        }

        public async Task<string> DeleteTestTemplate(string login, string testNumberIdentifier, string webRootPath)
        {
            TestTemplate testTemplate = GetTestTemplateDbSet().First(t => t.OwnerLogin == login && t.TestNumberIdentifier == testNumberIdentifier);
            return await dataFunctions.DeleteTestTemplate(testTemplate, webRootPath);
        }

        public IQueryable<QuestionTemplate> GetQuestionTemplates(string login, string testNumberIdentifier)
        {
             return GetQuestionTemplateDbSet()
                 .Include(q => q.SubquestionTemplateList)
                 .Include(q => q.TestTemplate)
                 .ThenInclude(q => q.Subject)
                 .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier && q.OwnerLogin == login).AsQueryable();
        }

        public QuestionTemplate GetQuestionTemplate(string login, string questionNumberIdentifier)
        {
            return dataFunctions.GetQuestionTemplate(login, questionNumberIdentifier);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            questionTemplate.QuestionNameIdentifier = "qna";
            questionTemplate.QuestionNumberIdentifier = "qnu";
            questionTemplate.Label = "temp";
            questionTemplate.SubquestionTemplateList = new List<SubquestionTemplate>();

            return await dataFunctions.AddQuestionTemplate(questionTemplate);
        }

        public async Task<string> DeleteQuestionTemplate(string login, string questionNumberIdentifier, string webRootPath)
        {
            return await dataFunctions.DeleteQuestionTemplate(login, questionNumberIdentifier, webRootPath);
        }

        public IQueryable<SubquestionTemplate> GetSubquestionTemplates(string login, string questionNumberIdentifier)
        {
            return GetSubquestionTemplateDbSet()
                .Include(s => s.QuestionTemplate)
                .Include(s => s.QuestionTemplate.TestTemplate)
                .Where(s => s.QuestionNumberIdentifier == questionNumberIdentifier && s.OwnerLogin == login).AsQueryable();
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            return await dataFunctions.AddSubquestionTemplate(subquestionTemplate, image, webRootPath);
        }

        public async Task<string> EditSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            string message;
            try
            {
                SubquestionTemplate oldSubquestionTemplate = GetSubquestionTemplate(subquestionTemplate.OwnerLogin, subquestionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier);
                oldSubquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText;
                oldSubquestionTemplate.PossibleAnswerList = subquestionTemplate.PossibleAnswerList;
                oldSubquestionTemplate.CorrectAnswerList = subquestionTemplate.CorrectAnswerList;
                oldSubquestionTemplate.SubquestionPoints = subquestionTemplate.SubquestionPoints;
                oldSubquestionTemplate.CorrectChoicePoints = subquestionTemplate.CorrectChoicePoints;
                oldSubquestionTemplate.DefaultWrongChoicePoints = subquestionTemplate.DefaultWrongChoicePoints;
                oldSubquestionTemplate.WrongChoicePoints = subquestionTemplate.WrongChoicePoints;

                bool subquestionContainedImage = false;
                string oldImagePath = "";
                if ((image != null || subquestionTemplate.ImageSource == null) && oldSubquestionTemplate.ImageSource != null)
                {
                    oldImagePath = oldSubquestionTemplate.ImageSource;
                    oldSubquestionTemplate.ImageSource = null;
                    subquestionContainedImage = true;
                }
                string newFileName = "";
                if (image != null)
                {
                    newFileName = Guid.NewGuid().ToString() + "_" + image.FileName;
                    oldSubquestionTemplate.ImageSource = newFileName;
                }
                await dataFunctions.SaveChangesAsync();
                message = "Zadání podotázky bylo úspěšně upraveno.";

                //only edit images in case all validity checks have already been passed
                if ((image != null || subquestionTemplate.ImageSource == null) && subquestionContainedImage)
                {
                    dataFunctions.DeleteSubquestionTemplateImage(webRootPath, oldImagePath);
                }
                if(image != null)
                {
                    dataFunctions.SaveImage(image, webRootPath, newFileName);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při úpravě podotázky nastala neočekávaná chyba.";
            }
            return message;
        }

        /// <summary>
        /// Validates the integrity of added subquestion template and changes certain fields before adding
        /// </summary>
        public (SubquestionTemplate, string?) ValidateSubquestionTemplate(SubquestionTemplate subquestionTemplate, string[] subquestionTextArray, string sliderValues,
            IFormFile? image)
        {
            string? errorMessage = null;

            //validate image
            if(image != null)
            {
                if (image.Length > 4000000)
                {
                    return(subquestionTemplate, "Chyba: Maximální povolená velikost obrázku je 4 MB.");
                }
                string fileExtension = Path.GetExtension(image.FileName).ToLower();
                if (fileExtension != ".jpg"
                    && fileExtension != ".png"
                    && fileExtension != ".jpeg"
                    && fileExtension != ".webp")
                {
                    return (subquestionTemplate, "Chyba: nepovolený formát (" + fileExtension + "). Povolené formáty jsou .jpg, .png, .jpeg, .webp");
                }
            }

            //remove any separators that are used in the database from the subquestion text
            if(subquestionTextArray.Length > 0)
            {
                subquestionTemplate.SubquestionText = "";
                for (int i = 0; i < subquestionTextArray.Length; i++)
                {
                    subquestionTextArray[i].Replace("|", "");//replace gap separator
                    subquestionTextArray[i].Replace(";", "");//replace answer separator
                    if (i != subquestionTextArray.Length - 1)
                    {
                        subquestionTemplate.SubquestionText += subquestionTextArray[i] + "|";
                    }
                    else
                    {
                        subquestionTemplate.SubquestionText += subquestionTextArray[i];
                    }
                }
            }
            else
            {
                subquestionTemplate.SubquestionText.Replace("|", "");//replace gap separator
                subquestionTemplate.SubquestionText.Replace(";", "");//replace answer separator
            }

            //change possible and correct answer lists if necessary
            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType.MatchingElements:
                    string[] correctAnswerList = new string[(subquestionTemplate.CorrectAnswerList.Length) / 2];
                    for(int i = 0; i < subquestionTemplate.CorrectAnswerList.Length; i++)
                    {
                        if(i % 2 == 1)
                        {
                            continue;
                        }
                        else
                        {
                            int index = i / 2;
                            correctAnswerList[index] = subquestionTemplate.CorrectAnswerList[i] + "|" + subquestionTemplate.CorrectAnswerList[i + 1];
                        }
                    }
                    subquestionTemplate.CorrectAnswerList = correctAnswerList;
                    break;
                case SubquestionType.FreeAnswer:
                    subquestionTemplate.PossibleAnswerList = new string[0];
                    subquestionTemplate.CorrectAnswerList = new string[0];
                    break;
                case SubquestionType.FreeAnswerWithDeterminedCorrectAnswer:
                    subquestionTemplate.PossibleAnswerList = new string[0];
                    break;
                case SubquestionType.GapMatch:
                    subquestionTemplate.PossibleAnswerList = new string[0];
                    break;
                case SubquestionType.Slider:
                    string[] sliderValuesSplit = sliderValues.Split(",");
                    subquestionTemplate.PossibleAnswerList = new string[] { sliderValuesSplit[0], sliderValuesSplit[1] };
                    subquestionTemplate.CorrectAnswerList = new string[] { sliderValuesSplit[2] };
                    break;
            }

            //todo: remove convert
            //set choice points
            if(subquestionTemplate.SubquestionPoints != null || subquestionTemplate.SubquestionPoints <= 0)
            {
                double correctChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(Convert.ToDouble(subquestionTemplate.SubquestionPoints),
                    subquestionTemplate.CorrectAnswerList, subquestionTemplate.SubquestionType);
                subquestionTemplate.CorrectChoicePoints = correctChoicePoints;
                subquestionTemplate.DefaultWrongChoicePoints = correctChoicePoints * (-1);
                if (subquestionTemplate.WrongChoicePoints != subquestionTemplate.DefaultWrongChoicePoints)//user manually set different wrong choice points than default
                {
                    if (subquestionTemplate.WrongChoicePoints < subquestionTemplate.DefaultWrongChoicePoints)//invalid value - cannot be lesser than default
                    {
                        subquestionTemplate.WrongChoicePoints = subquestionTemplate.DefaultWrongChoicePoints;
                    }
                }
            }
            else
            {
                errorMessage = "Chyba: nekompletní zadání podotázky (body).";
            }

            //validate integrity of the subquestion regarding amount of possible/correct answers, subquestion type and subquestion text content
            if(subquestionTemplate.SubquestionText.Length == 0)
            {
                errorMessage = "Chyba: nekompletní zadání podotázky (text).";
            }

            if(subquestionTemplate.SubquestionType == SubquestionType.Error)
            {
                errorMessage = "Chyba: nekompletní zadání podotázky (typ podotázky).";
            }

            for(int i = 0; i < subquestionTemplate.PossibleAnswerList.Length; i++)
            {
                if (subquestionTemplate.PossibleAnswerList[i].Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (možná odpověď).";
                }
            }

            for (int i = 0; i < subquestionTemplate.CorrectAnswerList.Length; i++)
            {
                if (subquestionTemplate.CorrectAnswerList[i].Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (správná odpověď).";
                }
            }

            if (subquestionTemplate.PossibleAnswerList.Distinct().Count() != subquestionTemplate.PossibleAnswerList.Count())
            {
                errorMessage = "Chyba: duplikátní možná odpověď.";
            }

            if (subquestionTemplate.CorrectAnswerList.Distinct().Count() != subquestionTemplate.CorrectAnswerList.Count() &&
                subquestionTemplate.SubquestionType != SubquestionType.MultipleQuestions)
            {
                errorMessage = "Chyba: duplikátní správná odpověď.";
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers
                 || subquestionTemplate.SubquestionType == SubquestionType.MatchingElements || subquestionTemplate.SubquestionType == SubquestionType.MultipleQuestions
                  || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill
                   || subquestionTemplate.SubquestionType == SubquestionType.Slider)
            {
                if (subquestionTemplate.PossibleAnswerList.Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (možné odpovědi).";
                }
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers
                 || subquestionTemplate.SubquestionType == SubquestionType.MatchingElements || subquestionTemplate.SubquestionType == SubquestionType.MultipleQuestions
                  || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill
                   || subquestionTemplate.SubquestionType == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.GapMatch
                    || subquestionTemplate.SubquestionType == SubquestionType.Slider)
            {
                if (subquestionTemplate.CorrectAnswerList.Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (správné odpovědi).";
                }
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers
                 || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill)
            {
                for(int i = 0; i < subquestionTemplate.CorrectAnswerList.Length; i++)
                {
                    if (!subquestionTemplate.PossibleAnswerList.Contains(subquestionTemplate.CorrectAnswerList[i]))
                    {
                        errorMessage = "Chyba: nekompletní zadání podotázky (možné a správné odpovědi).";
                    }
                }
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements)
            {
                if(subquestionTemplate.PossibleAnswerList.Length != subquestionTemplate.CorrectAnswerList.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers)
            {
                if (subquestionTemplate.PossibleAnswerList.Length < subquestionTemplate.CorrectAnswerList.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MatchingElements)
            {
                if (subquestionTemplate.PossibleAnswerList.Length / 2 < subquestionTemplate.CorrectAnswerList.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultipleQuestions)
            {
                if (subquestionTemplate.PossibleAnswerList.Length != subquestionTemplate.CorrectAnswerList.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.FreeAnswer)
            {
                if (subquestionTemplate.PossibleAnswerList.Length > 0 || subquestionTemplate.CorrectAnswerList.Length > 0)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer)
            {
                if (subquestionTemplate.PossibleAnswerList.Length <= 1 || subquestionTemplate.CorrectAnswerList.Length > 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill)
            {
                if (subquestionTemplate.PossibleAnswerList.Length <= 1 || subquestionTemplate.CorrectAnswerList.Length > 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
            {
                string[] subquestionTextGapSplit = subquestionTemplate.SubquestionText.Split("|");
                if (subquestionTextGapSplit.Length > 2)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
                if (subquestionTemplate.PossibleAnswerList.Length > 0 || subquestionTemplate.CorrectAnswerList.Length > 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.GapMatch)
            {
                string[] subquestionTextGapSplit = subquestionTemplate.SubquestionText.Split("|");
                if (subquestionTemplate.PossibleAnswerList.Length > 0 || subquestionTemplate.CorrectAnswerList.Length != subquestionTextGapSplit.Length - 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.Slider)
            {
                if (subquestionTemplate.PossibleAnswerList.Length != 2 || subquestionTemplate.CorrectAnswerList.Length != 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
                int lowerBound = int.Parse(subquestionTemplate.PossibleAnswerList[0]);
                int upperBound = int.Parse(subquestionTemplate.PossibleAnswerList[1]);
                int answer = int.Parse(subquestionTemplate.CorrectAnswerList[0]);
                if (answer < lowerBound || answer > upperBound || lowerBound > upperBound)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (hodnoty posuvníku).";
                }
            }

            return (subquestionTemplate, errorMessage);
        }

        /// <summary>
        /// Changes certain fields of subquestion template before finally sending them to the presentation layer
        /// </summary>
        public List<SubquestionTemplate> ProcessSubquestionTemplateForView(List<SubquestionTemplate> subquestionTemplates)
        {
            List<SubquestionTemplate> processedSubquestionTemplates = new List<SubquestionTemplate>();
            string[] possibleAnswerList;
            string[] correctAnswerList;

            foreach (SubquestionTemplate subquestionTemplate in subquestionTemplates)
            {
                switch (subquestionTemplate.SubquestionType)
                {
                    case SubquestionType.MatchingElements:
                        correctAnswerList = new string[subquestionTemplate.CorrectAnswerList.Length];
                        for(int i = 0; i < subquestionTemplate.CorrectAnswerList.Length; i++)
                        {
                            correctAnswerList[i] = subquestionTemplate.CorrectAnswerList[i].Replace("|", " -> ");
                        }
                        subquestionTemplate.CorrectAnswerList = correctAnswerList;
                        break;
                    case SubquestionType.MultipleQuestions:
                        possibleAnswerList = new string[2] { "Ano", "Ne" };
                        correctAnswerList = new string[subquestionTemplate.PossibleAnswerList.Length];
                        for (int i = 0; i < subquestionTemplate.PossibleAnswerList.Length; i++)
                        {
                            string answer = "";
                            if (subquestionTemplate.CorrectAnswerList[i] == "1")
                            {
                                answer = "Ano";
                            }
                            else if (subquestionTemplate.CorrectAnswerList[i] == "0")
                            {
                                answer = "Ne";
                            }
                            correctAnswerList[i] = subquestionTemplate.PossibleAnswerList[i] + " -> " + answer;
                        }
                        subquestionTemplate.PossibleAnswerList = possibleAnswerList;
                        subquestionTemplate.CorrectAnswerList = correctAnswerList;
                        break;
                    case SubquestionType.MultiChoiceTextFill:
                        subquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText.Replace("|", " (DOPLŇTE) ");
                        break;
                    case SubquestionType.FreeAnswerWithDeterminedCorrectAnswer:
                        subquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText.Replace("|", " (DOPLŇTE) ");
                        break;
                    case SubquestionType.GapMatch:
                        string[] subquestionTextSplit = subquestionTemplate.SubquestionText.Split('|');
                        subquestionTemplate.SubquestionText = "";
                        for (int i = 0; i < subquestionTextSplit.Length; i++)
                        {
                            if (i != subquestionTextSplit.Length - 1)
                            {
                                subquestionTemplate.SubquestionText += subquestionTextSplit[i] + " (DOPLŇTE[" + (i + 1) + "]) ";
                            }
                            else
                            {
                                subquestionTemplate.SubquestionText += subquestionTextSplit[i];
                            }
                        }

                        subquestionTemplate.PossibleAnswerList = subquestionTemplate.CorrectAnswerList;
                        correctAnswerList = new string[subquestionTemplate.CorrectAnswerList.Length];
                        for (int i = 0; i < subquestionTemplate.CorrectAnswerList.Length; i++)
                        {
                            correctAnswerList[i] = "[" + (i + 1) + "] - " + subquestionTemplate.CorrectAnswerList[i];
                        }
                        subquestionTemplate.CorrectAnswerList = correctAnswerList;
                        break;
                    case SubquestionType.Slider:
                        possibleAnswerList = new string[] { subquestionTemplate.PossibleAnswerList[0] + " - " + subquestionTemplate.PossibleAnswerList[1] };
                        subquestionTemplate.PossibleAnswerList = possibleAnswerList;
                        break;
                }

                processedSubquestionTemplates.Add(subquestionTemplate);
            }


            return processedSubquestionTemplates;
        }

        public async Task<string> DeleteSubquestionTemplate(string login, string questionNumberIdentifier, string subquestionIdentifier, string webRootPath)
        {
            return await dataFunctions.DeleteSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier, webRootPath);
        }

        public TestTemplate GetTestTemplate(string testNumberIdentifier)
        {
            return GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.Owner)
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .First(t => t.TestNumberIdentifier == testNumberIdentifier);
        }

        public TestTemplate GetTestTemplate(string login, string testNumberIdentifier)
        {
            return GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .First(t => t.TestNumberIdentifier == testNumberIdentifier && t.OwnerLogin == login);
        }

        public SubquestionTemplate GetSubquestionTemplate(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return GetSubquestionTemplateDbSet()
                .First(s => s.QuestionNumberIdentifier == questionNumberIdentifier
                && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
        }

        public SubquestionTemplateStatistics? GetSubquestionTemplateStatistics(string login)
        {
            return dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefault(s => s.UserLogin == login);
        }

        public TestDifficultyStatistics? GetTestDifficultyStatistics(string login)
        {
            return dataFunctions.GetTestDifficultyStatisticsDbSet().FirstOrDefault(s => s.UserLogin == login);
        }

        public async Task SetNegativePoints(TestTemplate testTemplate, EnumTypes.NegativePoints negativePoints)
        {
            testTemplate.NegativePoints = negativePoints;
            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> SetMinimumPoints(TestTemplate testTemplate, double minimumPoints, string testPointsDetermined)
        {
            string message = string.Empty;
            double? totalTestPoints = GetTestTemplatePointsSum(testTemplate);
            if (minimumPoints < 0 || minimumPoints > totalTestPoints)
            {
                double? totalTestPointsRound = CommonFunctions.RoundDecimal(totalTestPoints);
                message = "Chyba: Hodnota musí být mezi 0 a " + totalTestPointsRound.ToString();
            }
            else if (testPointsDetermined == "False")//todo: parse to bool
            {
                message = "Chyba: Nejprve je nutné nastavit body u všech otázek testu.";
            }
            else
            {
                testTemplate.MinimumPoints = minimumPoints;
                message = "Změny úspěšně uloženy.";
                await dataFunctions.SaveChangesAsync();
            }
            return message;
        }

        public async Task<string> SetSubquestionTemplatePoints(string login, string questionNumberIdentifier, string subquestionIdentifier, string subquestionPoints, string wrongChoicePoints, bool defaultWrongChoicePoints)
        {
            string message = string.Empty;
            var subquestionTemplate = GetSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier);
            if (subquestionTemplate != null)
            {
                //user has attached points to a subquestion that did not have them before - increment SubquestionTemplatesAdded variable
                if (subquestionPoints != null && subquestionTemplate.SubquestionPoints == null)
                {
                    User user = dataFunctions.GetUserByLogin(login);
                    SubquestionTemplateStatistics subquestionTemplateStatistics = GetSubquestionTemplateStatistics(login);
                    subquestionTemplateStatistics.SubquestionTemplatesAdded++;
                }

                double? subquestionPointsDouble = null;
                if (subquestionPoints != null)
                {
                    subquestionPointsDouble = Math.Round(double.Parse(subquestionPoints, CultureInfo.InvariantCulture), 2);
                }
                double? wrongChoicePointsDouble = null;
                if(wrongChoicePoints != null)
                {
                    wrongChoicePointsDouble = Math.Round(double.Parse(wrongChoicePoints, CultureInfo.InvariantCulture), 2);
                }

                if (subquestionPoints == null)
                {
                    message = "Chyba: nebyl zadán žádný počet bodů.";
                }
                else if (!double.TryParse(subquestionPoints, out _))
                {
                    message = "Chyba: \"" + subquestionPoints + "\" není korektní formát počtu bodů. Je nutné zadat číslo.";
                }
                else if (subquestionPointsDouble <= 0)
                {
                    message = "Chyba: otázce je nutné přidělit kladný počet bodů.";
                }
                else if (!defaultWrongChoicePoints && wrongChoicePointsDouble * (-1) > subquestionPointsDouble)
                {
                    message = "Chyba: za špatnou volbu nemůže student obdržet méně než " + subquestionPointsDouble * (-1) + " bodů.";
                }
                else//todo: overit jestli nema za otazku nektery student pridelen vyssi pocet bodu nez soucasny pocet bodu
                {
                    message = "Počet bodů byl úspěšně změněn.";
                    subquestionTemplate.SubquestionPoints = Math.Round(Convert.ToDouble(subquestionPoints), 2);
                    subquestionTemplate.CorrectChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(
                        Math.Round(Convert.ToDouble(subquestionPoints), 2), subquestionTemplate.CorrectAnswerList, subquestionTemplate.SubquestionType);

                    if(subquestionTemplate.WrongChoicePoints == null)
                    {
                        subquestionTemplate.DefaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                        subquestionTemplate.WrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                    }
                    else
                    {
                        if(defaultWrongChoicePoints)
                        {
                            subquestionTemplate.DefaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                            subquestionTemplate.WrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                        }
                        else
                        {
                            subquestionTemplate.DefaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                            subquestionTemplate.WrongChoicePoints = Math.Round(Convert.ToDouble(wrongChoicePoints, CultureInfo.InvariantCulture), 2);
                        }
                    }
                    await dataFunctions.SaveChangesAsync();
                }
            }
            return message;
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            User owner = dataFunctions.GetUserByLogin(login);

            //check if enough subquestion templates have been added to warrant new model training
            bool retrainModel = false;
            int subquestionTemplatesAdded = GetSubquestionTemplateStatistics(login).SubquestionTemplatesAdded;
            if (subquestionTemplatesAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionTemplateModel(owner);
            }

            var subquestionTemplates = GetSubquestionTemplates(login, questionNumberIdentifier);

            if (subquestionIdentifier == null)
            {
                subquestionIdentifier = subquestionTemplates.First().SubquestionIdentifier;
            }

            var subquestionTemplate = GetSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier);
            var testTemplates = dataFunctions.GetTestTemplateList(owner.Login);
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageTemplatePoints(testTemplates);
            double[] subjectAveragePoints = DataGenerator.GetSubjectAverageTemplatePoints(testTemplates);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double? minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionTemplateRecord currentSubquestionTemplateRecord = DataGenerator.CreateSubquestionTemplateRecord(subquestionTemplate, owner, subjectsArray,
                subquestionTypeAveragePoints, subjectAveragePoints, minimumPointsShare);
            SubquestionTemplateStatistics? currectSubquestionTemplateStatistics = GetSubquestionTemplateStatistics(login);
            Model usedModel = currectSubquestionTemplateStatistics.UsedModel;
            string suggestedSubquestionPoints = PythonFunctions.GetSubquestionTemplateSuggestedPoints(login, retrainModel, currentSubquestionTemplateRecord, usedModel);
            if (subquestionTemplatesAdded >= 100)
            {
                SubquestionTemplateStatistics subquestionTemplateStatistics = GetSubquestionTemplateStatistics(login);
                subquestionTemplateStatistics.SubquestionTemplatesAdded = 0;
                subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateNeuralNetwork.py");
                subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateMachineLearning.py");
                if(subquestionTemplateStatistics.NeuralNetworkAccuracy >= subquestionTemplateStatistics.MachineLearningAccuracy)
                {
                    subquestionTemplateStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionTemplateStatistics.UsedModel = Model.MachineLearning;
                }
                await dataFunctions.SaveChangesAsync();
            }

            return suggestedSubquestionPoints;
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(SubquestionTemplate subquestionTemplate)
        {
            //TODO: Uprava funkce - pouziti pouze 1 funkce
            string login = subquestionTemplate.OwnerLogin;
            User owner = dataFunctions.GetUserByLogin(login);

            //check if enough subquestion templates have been added to warrant new model training
            bool retrainModel = false;
            int subquestionTemplatesAdded = GetSubquestionTemplateStatistics(subquestionTemplate.OwnerLogin).SubquestionTemplatesAdded;
            if (subquestionTemplatesAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionTemplateModel(owner);
            }

      /*      var subquestionTemplates = GetSubquestionTemplates(login, questionNumberIdentifier);

            if (subquestionIdentifier == null)
            {
                subquestionIdentifier = subquestionTemplates.First().SubquestionIdentifier;
            }

            var subquestionTemplate = GetSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier);*/
            var testTemplates = dataFunctions.GetTestTemplateList(owner.Login);
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageTemplatePoints(testTemplates);
            double[] subjectAveragePoints = DataGenerator.GetSubjectAverageTemplatePoints(testTemplates);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double? minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionTemplateRecord currentSubquestionTemplateRecord = DataGenerator.CreateSubquestionTemplateRecord(subquestionTemplate, owner, subjectsArray,
                subquestionTypeAveragePoints, subjectAveragePoints, minimumPointsShare);
            SubquestionTemplateStatistics? currectSubquestionTemplateStatistics = GetSubquestionTemplateStatistics(login);
            Model usedModel = currectSubquestionTemplateStatistics.UsedModel;
            string suggestedSubquestionPoints = PythonFunctions.GetSubquestionTemplateSuggestedPoints(login, retrainModel, currentSubquestionTemplateRecord, usedModel);
            if (subquestionTemplatesAdded >= 100)
            {
                SubquestionTemplateStatistics subquestionTemplateStatistics = GetSubquestionTemplateStatistics(login);
                subquestionTemplateStatistics.SubquestionTemplatesAdded = 0;
                subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateNeuralNetwork.py");
                subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateMachineLearning.py");
                if (subquestionTemplateStatistics.NeuralNetworkAccuracy >= subquestionTemplateStatistics.MachineLearningAccuracy)
                {
                    subquestionTemplateStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionTemplateStatistics.UsedModel = Model.MachineLearning;
                }
                await dataFunctions.SaveChangesAsync();
            }

            return suggestedSubquestionPoints;
        }

        public async Task RetrainSubquestionTemplateModel(User owner)
        {
            string login = owner.Login;
            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testTemplates = dataFunctions.GetTestTemplateList(login);

            var subquestionTemplateRecords = DataGenerator.CreateSubquestionTemplateRecords(testTemplates);
            await dataFunctions.SaveSubquestionTemplateRecords(subquestionTemplateRecords, owner);
        }

        public List<TestTemplate> GetTestingDataTestTemplates()
        {
            var testTemplates = GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .Where(t => t.IsTestingData).ToList();
            return testTemplates;
        }

        public int GetTestingDataSubquestionTemplatesCount()
        {
            int testingDataSubquestionTemplates = 0;
            var testTemplates = GetTestingDataTestTemplates();
            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];
                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);
                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        testingDataSubquestionTemplates++;
                    }
                }
            }
            return testingDataSubquestionTemplates;
        }

        public async Task<string> CreateTemplateTestingData(string action, string amountOfSubquestionTemplates)
        {
            string message;
            await TestingUsersCheck();
            User? owner = dataFunctions.GetUserByLogin("login");
            var existingTestTemplates = GetTestingDataTestTemplates();

            List<TestTemplate> testTemplates = new List<TestTemplate>();
            if (action == "addSubquestionTemplateRandomData")
            {
                testTemplates = DataGenerator.GenerateRandomTestTemplates(existingTestTemplates, Convert.ToInt32(amountOfSubquestionTemplates));
            }
            else if (action == "addSubquestionTemplateCorrelationalData")
            {
                testTemplates = DataGenerator.GenerateCorrelationalTestTemplates(existingTestTemplates, Convert.ToInt32(amountOfSubquestionTemplates));
            }
            message = await dataFunctions.AddTestTemplates(testTemplates, owner);//todo: error?
            string login = "login";
            owner = dataFunctions.GetUserByLoginAsNoTracking();

            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testTemplatesToRecord = dataFunctions.GetTestTemplateList(login);

            var subquestionTemplateRecords = DataGenerator.CreateSubquestionTemplateRecords(testTemplatesToRecord);
            await dataFunctions.SaveSubquestionTemplateRecords(subquestionTemplateRecords, owner);

            dataFunctions.ClearChargeTracker();
            owner = dataFunctions.GetUserByLoginAsNoTracking();
            var subquestionTemplateStatistics = GetSubquestionTemplateStatistics(owner.Login);
            if (subquestionTemplateStatistics == null)
            {
                subquestionTemplateStatistics = new SubquestionTemplateStatistics();
                subquestionTemplateStatistics.User = owner;
                subquestionTemplateStatistics.UserLogin = owner.Login;
                subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "TemplateNeuralNetwork.py");
                subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateMachineLearning.py");
                if (subquestionTemplateStatistics.NeuralNetworkAccuracy >= subquestionTemplateStatistics.MachineLearningAccuracy)
                {
                    subquestionTemplateStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionTemplateStatistics.UsedModel = Model.MachineLearning;
                }
                await dataFunctions.AddSubquestionTemplateStatistics(subquestionTemplateStatistics);
                dataFunctions.AttachUser(subquestionTemplateStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }
            else
            {
                subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "TemplateNeuralNetwork.py");
                subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateMachineLearning.py");
                if (subquestionTemplateStatistics.NeuralNetworkAccuracy >= subquestionTemplateStatistics.MachineLearningAccuracy)
                {
                    subquestionTemplateStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionTemplateStatistics.UsedModel = Model.MachineLearning;
                }
                await dataFunctions.SaveChangesAsync();
            }

            return message;
        }

        public async Task TestingUsersCheck()
        {
            User? owner = dataFunctions.GetUserByLogin("login");
            if(owner == null)
            {
                owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
                await dataFunctions.AddUser(owner);
            }
        }

        public async Task DeleteTemplateTestingData()
        {
            dataFunctions.ExecuteSqlRaw("delete from TestTemplate where IsTestingData = 1");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateRecord where OwnerLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateStatistics where UserLogin = 'login'");

            //since results are directly linked to templates, they are deleted as well
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where OwnerLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultStatistics where UserLogin = 'login'");
            await dataFunctions.SaveChangesAsync();
        }

        public string GetTestDifficultyPrediction(string login, string testNumberIdentifier)
        {
            string testDifficultyMessage;
            SubquestionResultStatistics? subquestionResultStatistics = dataFunctions.GetSubquestionResultStatistics(login);
            //check whether there are enough result statistics to go by
            if(subquestionResultStatistics == null)
            {
                return "Chyba: nedostatečný počet výsledků testů.;";
            }

            //predicted amount of points that the average student will get for this test
            double testTemplatePredictedPoints = PythonFunctions.GetTestTemplatePredictedPoints(false, login, subquestionResultStatistics.UsedModel, testNumberIdentifier);

            List<TestResult> testResults = dataFunctions.GetTestResultsByLogin(login);
            List<(string, int, double?)> testResultsPointsShare = new List<(string, int, double?)>();
            
            //iterate through every test result to obtain the share between student's points and test total points
            for(int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];
                TestTemplate testTemplate = testResult.TestTemplate;

                //skip the test that's currently being predicted as we want it to compare to other tests
                if(testTemplate.TestNumberIdentifier == testNumberIdentifier)
                {
                    continue;
                }

                double? testTemplatePointsSum = GetTestTemplatePointsSum(testTemplate);
                double? studentsPoints = 0;

                for (int j = 0; j < testResult.QuestionResultList.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(j);

                    for(int k = 0; k < questionResult.SubquestionResultList.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(k);
                        studentsPoints += subquestionResult.StudentsPoints;
                    }
                }

                bool matchFound = false;
                for(int l = 0; l < testResultsPointsShare.Count; l++)
                {
                    //record of this test template is already in the testResultPointsShare list - we modify it
                    if (testResultsPointsShare[l].Item1 == testTemplate.TestNumberIdentifier)
                    {
                        matchFound = true;
                        testResultsPointsShare[l] = (testResultsPointsShare[l].Item1, testResultsPointsShare[l].Item2 + 1, testResultsPointsShare[l].Item3 + studentsPoints);
                    }
                }

                //record of this test template is not in the testResultPointsShare list - we add a new one
                if (!matchFound)
                {
                    testResultsPointsShare.Add((testTemplate.TestNumberIdentifier, 1, studentsPoints / testTemplatePointsSum));
                }
            }

            double?[] testTemplatesPointsShare = new double?[testResultsPointsShare.Count];
            for (int i = 0; i < testResultsPointsShare.Count; i++)
            {
                testTemplatesPointsShare[i] = testResultsPointsShare[i].Item3 / testResultsPointsShare[i].Item2;
            }
            TestTemplate currentTestTemplate = GetTestTemplate(login, testNumberIdentifier);
            double? currentTestTemplatePointsShare = testTemplatePredictedPoints / GetTestTemplatePointsSum(currentTestTemplate);
            //compare the ratio of predicted test points to total test points with the average points of all test templates (as measured by existing test results)
            double? difficulty = currentTestTemplatePointsShare / testTemplatesPointsShare.Average();

            testDifficultyMessage = testTemplatePredictedPoints + ";";
            if (difficulty < 1)//easier than the average test
            {
                difficulty *= 100;
                difficulty = Math.Round(difficulty.Value, 0);
                testDifficultyMessage += "Test je o " + (100 - difficulty) + "% lehčí než průměrný test.";
            }
            else//harder than the average test
            {
                difficulty -= 1;
                difficulty *= 100;
                difficulty = Math.Round(difficulty.Value, 0);
                testDifficultyMessage += "Test je o " + difficulty + "% těžší než průměrný test.";
            }

            return testDifficultyMessage;
        }

        public double? GetTestTemplatePointsSum(TestTemplate testTemplate)
        {
            double? testPoints = 0;
            for (int i = 0; i < testTemplate.QuestionTemplateList.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(i);

                for (int j = 0; j < questionTemplate.SubquestionTemplateList.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(j);
                    testPoints += subquestionTemplate.SubquestionPoints;
                }
            }

            return testPoints;
        }

        public int GetTestTemplateSubquestionsCount(TestTemplate testTemplate)
        {
            int subquestionsCount = 0;
            for (int i = 0; i < testTemplate.QuestionTemplateList.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(i);

                for (int j = 0; j < questionTemplate.SubquestionTemplateList.Count; j++)
                {
                    subquestionsCount++;
                }
            }

            return subquestionsCount;
        }

        /// <summary>
        /// Returns the list of test templates
        /// </summary>
        public List<TestTemplate> LoadTestTemplates(string login)
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            string subDirectory = "";

            if (Directory.Exists(Config.GetTestTemplatesPath()))
            {
                foreach (var directory in Directory.GetDirectories(Config.GetTestTemplatesPath()))
                {
                    string[] splitDirectoryBySlash = directory.Split(Config.GetPathSeparator());
                    string testNameIdentifier = splitDirectoryBySlash[splitDirectoryBySlash.Length - 1].ToString();
                    string testNumberIdentifier = "";

                    try
                    {
                        foreach (var directory_ in Directory.GetDirectories(directory + Config.GetPathSeparator() + "tests"))
                        {
                            string[] splitDirectory_BySlash = directory_.Split(Config.GetPathSeparator());
                            testNumberIdentifier = splitDirectory_BySlash[splitDirectory_BySlash.Length - 1].ToString();
                            subDirectory = directory_;
                        }
                    }
                    catch
                    {
                        continue;
                    }

                    try
                    {
                        XmlReader xmlReader = XmlReader.Create(subDirectory + Config.GetPathSeparator() + "test.xml");
                        while (xmlReader.Read())
                        {
                            if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                            {
                                if (xmlReader.HasAttributes)
                                {
                                    TestTemplate testTemplate = new TestTemplate();
                                    testTemplate.TestNameIdentifier = testNameIdentifier;
                                    testTemplate.TestNumberIdentifier = testNumberIdentifier;
                                    if (xmlReader.GetAttribute("title") != null)
                                    {
                                        testTemplate.Title = xmlReader.GetAttribute("title")!;
                                    }
                                    testTemplate.OwnerLogin = login;
                                    if (dataFunctions.GetUserByLogin(login) != null)
                                    {
                                        testTemplate.Owner = dataFunctions.GetUserByLogin(login);
                                    }
                                    else
                                    {
                                        throw Exceptions.SpecificUserNotFoundException(login);
                                    }
                                    testTemplate.QuestionTemplateList = LoadQuestionTemplates(testTemplate, login);
                                    testTemplates.Add(testTemplate);
                                }
                            }
                        }
                    }
                    catch
                    {
                        continue;
                    }
                }
                return testTemplates;
            }
            else
            {
                throw Exceptions.TestTemplatesPathNotFoundException;
            }
        }

        /// <summary>
        /// Returns the list of question templates (of a particular test template) with all their parameters
        /// </summary>
        /// <param name="testTemplate">Test template that the returned question templates belong to</param>
        /// <param name="login">Login of the staff member that's uploading these question templates</param>
        /// <returns>the list of question templates with all their parameters</returns>
        public List<QuestionTemplate> LoadQuestionTemplates(TestTemplate testTemplate, string login)
        {
            int i = 0;

            //this list won't be actually returned because its elements may not be in the correct order
            List<QuestionTemplate> questionTemplatesTemp = new List<QuestionTemplate>();

            //a separate function LoadQuestionParameters is used here because some of the question parameters are located in the test.xml file, while others are in the qti.xml file
            List<(string, string, string, string)> questionParameters = LoadQuestionParameters(testTemplate.TestNameIdentifier, testTemplate.TestNumberIdentifier);

            if (Directory.Exists(Config.GetQuestionTemplatesPath(testTemplate.TestNameIdentifier)))
            {
                foreach (var directory in Directory.GetDirectories(Config.GetQuestionTemplatesPath(testTemplate.TestNameIdentifier)))
                {
                    foreach (var file in Directory.GetFiles(directory))
                    {
                        string[] fileSplitBySlash = file.Split(Config.GetPathSeparator());
                        if (fileSplitBySlash[fileSplitBySlash.Length - 1] != "qti.xml")
                        {
                            continue;
                        }
                        else
                        {
                            XmlReader xmlReader = XmlReader.Create(file);
                            while (xmlReader.Read())
                            {
                                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                                {
                                    if (xmlReader.Name == "assessmentItem")
                                    {
                                        for (int j = 0; j < questionParameters.Count; j++)
                                        {
                                            if (questionParameters[j].Item4 == xmlReader.GetAttribute("identifier"))
                                            {
                                                QuestionTemplate questionTemplate = new QuestionTemplate();
                                                questionTemplate.QuestionNameIdentifier = questionParameters[j].Item3;
                                                questionTemplate.QuestionNumberIdentifier = questionParameters[j].Item4;
                                                if (xmlReader.GetAttribute("title") != null)
                                                {
                                                    questionTemplate.Title = xmlReader.GetAttribute("title")!;
                                                }
                                                if (xmlReader.GetAttribute("label") != null)
                                                {
                                                    questionTemplate.Label = xmlReader.GetAttribute("label")!;
                                                }
                                                questionTemplate.OwnerLogin = login;
                                                questionTemplate.TestTemplate = testTemplate;
                                                questionTemplate.SubquestionTemplateList = LoadSubquestionTemplates(testTemplate.TestNameIdentifier, questionTemplate, login);
                                                questionTemplatesTemp.Add(questionTemplate);
                                                i++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else { throw Exceptions.QuestionTemplatesPathNotFoundException(testTemplate.TestNameIdentifier); }

            //correction of potential wrong order of elements in the questionTemplatesTemp array
            List<QuestionTemplate> questionTemplates = new List<QuestionTemplate>();
            for (int k = 0; k < questionParameters.Count; k++)
            {
                for (int l = 0; l < questionTemplatesTemp.Count; l++)
                {
                    if (questionParameters[k].Item4 == questionTemplatesTemp[l].QuestionNumberIdentifier)
                    {
                        questionTemplates.Add(questionTemplatesTemp[l]);
                    }
                }
            }

            return questionTemplates;
        }

        /// <summary>
        /// Returns the list of questions with certain parameters - name/number identifiers, and test part/section they belong to from the test.xml file
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test</param>
        /// <param name="testNumberIdentifier">Number identifier of the selected test</param>
        /// <returns>the list of questions and their name/number identifiers, and test part/section they belong to</returns>
        public List<(string, string, string, string)> LoadQuestionParameters(string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string)> questionParameters = new List<(string, string, string, string)>();
            string testPart = "";
            string testSection = "";
            string questionNameIdentifier = "";
            string questionNumberIdentifier = "";
            string questionNumberIdentifierToSplit = "";

            XmlReader xmlReader = XmlReader.Create(Config.GetTestTemplateFilePath(testNameIdentifier, testNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testPart" && xmlReader.GetAttribute("identifier") != null)
                {
                    testPart = xmlReader.GetAttribute("identifier")!;
                }

                if (xmlReader.Name == "assessmentSection" && xmlReader.GetAttribute("identifier") != null)
                {
                    testSection = xmlReader.GetAttribute("identifier")!;
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    if (xmlReader.GetAttribute("identifier") != null)
                    {
                        questionNameIdentifier = xmlReader.GetAttribute("identifier")!;
                    }

                    if (xmlReader.GetAttribute("href") != null)
                    {
                        questionNumberIdentifierToSplit = xmlReader.GetAttribute("href")!;
                    }
                    string[] questionNumberIdentifierSplitBySlash = questionNumberIdentifierToSplit.Split(@"/");
                    questionNumberIdentifier = questionNumberIdentifierSplitBySlash[3];
                    questionParameters.Add((testPart, testSection, questionNameIdentifier, questionNumberIdentifier));
                }
            }

            return questionParameters;
        }

        /// <summary>
        /// Returns the list of all subquestion templates that are included in the selected question
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <returns>the list of all subquestion templates (by question template)</returns>
        public List<SubquestionTemplate> LoadSubquestionTemplates(string testNameIdentifier, QuestionTemplate questionTemplate, string login)
        {
            List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();
            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionTemplate.QuestionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                    if (xmlReader.GetAttribute("identifier") != null)
                    {
                        string subquestionIdentifier = xmlReader.GetAttribute("identifier")!;
                        subquestionTemplate.SubquestionIdentifier = subquestionIdentifier;

                        EnumTypes.SubquestionType subquestionType = GetSubquestionType(subquestionIdentifier, testNameIdentifier, questionTemplate.QuestionNumberIdentifier);
                        subquestionTemplate.SubquestionType = subquestionType;

                        subquestionTemplate.ImageSource = GetSubquestionImage(subquestionIdentifier, testNameIdentifier, questionTemplate.QuestionNumberIdentifier);

                        subquestionTemplate.PossibleAnswerList = GetPossibleAnswerList(subquestionIdentifier, testNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionType);

                        subquestionTemplate.CorrectAnswerList = GetCorrectAnswerList(subquestionIdentifier, testNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionType);

                        subquestionTemplate.SubquestionText = GetSubquestionText(subquestionIdentifier, testNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionType, subquestionTemplate.CorrectAnswerList.Length);

                        subquestionTemplate.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;

                        subquestionTemplate.OwnerLogin = login;

                        subquestionTemplate.QuestionTemplate = questionTemplate;

                        subquestionTemplates.Add(subquestionTemplate);
                    }
                }
            }
            return subquestionTemplates;
        }

        /// <summary>
        /// Returns the type of subquestion
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the type of subquestion</returns>
        public EnumTypes.SubquestionType GetSubquestionType(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            EnumTypes.SubquestionType subquestionType = EnumTypes.SubquestionType.Error;
            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            bool singleCorrectAnswer = false;//questionType = 6 nebo 7; jediná správná odpověď

            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    if (xmlReader.GetAttribute("identifier") != null)
                    {
                        string subquestionIdentifier = xmlReader.GetAttribute("identifier")!;

                        //skip other subquestions
                        if (subquestionIdentifier != null && subquestionIdentifier != selectedSubquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }

                        if (xmlReader.GetAttribute("cardinality") == "ordered" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            subquestionType = EnumTypes.SubquestionType.OrderingElements;//Typ otázky = seřazení pojmů
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            subquestionType = EnumTypes.SubquestionType.MultiChoiceMultipleCorrectAnswers;//Typ otázky = více odpovědí (abc); více odpovědí může být správně
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "pair")
                        {
                            subquestionType = EnumTypes.SubquestionType.MatchingElements;//Typ otázky = spojování párů
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                        {
                            subquestionType = EnumTypes.SubquestionType.MultipleQuestions;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                        {
                            subquestionType = EnumTypes.SubquestionType.FreeAnswer;//Typ otázky = volná odpověď; odpověď není předem daná
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "integer")
                        {
                            subquestionType = EnumTypes.SubquestionType.Slider;//Typ otázky = slider
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            singleCorrectAnswer = true;
                        }
                    }
                }

                //skip other subquestions - outside of response declaration
                var name = xmlReader.Name;
                if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                    name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                {
                    if (xmlReader.GetAttribute("responseIdentifier") != null)
                    {
                        string subquestionIdentifier = xmlReader.GetAttribute("responseIdentifier")!;
                        if (subquestionIdentifier != null && subquestionIdentifier != selectedSubquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                    }

                }

                if (xmlReader.Name == "gapMatchInteraction")
                {
                    if (xmlReader.GetAttribute("responseIdentifier") != null)
                    {
                        string subquestionIdentifier = xmlReader.GetAttribute("responseIdentifier")!;
                        if (subquestionIdentifier == selectedSubquestionIdentifier)
                        {
                            subquestionType = EnumTypes.SubquestionType.GapMatch;
                        }
                    }

                }

                if (singleCorrectAnswer)
                {
                    if (xmlReader.Name == "simpleChoice")
                    {
                        subquestionType = EnumTypes.SubquestionType.MultiChoiceSingleCorrectAnswer;//Typ otázky = výběr z více možností (abc), jen jedna odpověď je správně
                    }
                }

                if (xmlReader.Name == "textEntryInteraction" && subquestionType == EnumTypes.SubquestionType.FreeAnswer)
                {
                    subquestionType = EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer;//Typ otázky = volná odpověď; odpověď je předem daná
                }
            }

            if (singleCorrectAnswer && subquestionType == 0)
            {
                subquestionType = EnumTypes.SubquestionType.MultiChoiceTextFill;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
            }
            return subquestionType;
        }

        public string GetSubquestionTypeText(int subquestionType)
        {
            switch (subquestionType)
            {
                case 0:
                    return "Neznámý nebo nepodporovaný typ otázky!";
                case 1:
                    return "Seřazení pojmů";
                case 2:
                    return "Výběr z více možností; více možných správných odpovědí";
                case 3:
                    return "Spojování pojmů";
                case 4:
                    return "Více otázek k jednomu pojmu; více možných správných odpovědí";
                case 5:
                    return "Volná odpověď, správná odpověď není automaticky určena";
                case 6:
                    return "Výběr z více možností; jedna správná odpověd";
                case 7:
                    return "Výběr z více možností (doplnění textu); jedna správná odpověď";
                case 8:
                    return "Volná odpověď, správná odpověď je automaticky určena";
                case 9:
                    return "Dosazování pojmů do mezer";
                case 10:
                    return "Posuvník; jedna správná odpověď (číslo)";
                default:
                    return "Neznámý nebo nepodporovaný typ otázky!";
            }
        }

        public string[] SubquestionTypeTextArray = {
        "Neznámý nebo nepodporovaný typ otázky!",
        "Seřazení pojmů",
        "Výběr z více možností; více možných správných odpovědí",
        "Spojování pojmů",
        "Více otázek k jednomu pojmu; více možných správných odpovědí",
        "Volná odpověď, správná odpověď není automaticky určena",
        "Výběr z více možností; jedna správná odpověd",
        "Výběr z více možností (doplnění textu); jedna správná odpověď",
        "Volná odpověď, správná odpověď je automaticky určena",
        "Dosazování pojmů do mezer",
        "Posuvník; jedna správná odpověď (číslo)"};

        public string[] GetSubquestionTypeTextArray()
        {
            return SubquestionTypeTextArray;
        }

        /// <summary>
        /// Returns the subquestion text
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the text of subquestion</returns>
        public string GetSubquestionText(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, EnumTypes.SubquestionType subquestionType, int correctAnswersCount)
        {
            string subquestionText = "";
            string subquestionTextTemporary = "";
            int amountOfGaps = 0;//for subquestionType 9 (multiple gaps)
            int simpleMatchSetCounter = 0;//for subquestionType 4
            bool pTagVisited = false;
            bool divTagVisited = false;
            bool identifierCheck = false;//checks whether the current subquestion's identifier matches the "subquestionIdentifier" variable
            int nodeCount = 0;//counts how many nodes have been read by the XmlReader
            int oldNodeCount = 0;

            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                nodeCount++;
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            identifierCheck = false;
                            xmlReader.Skip();
                        }
                        else
                        {
                            identifierCheck = true;
                        }
                    }
                }

                //after the node with the subquestion's identifier has been read, it's necessary to set identifierCheck to false
                if (xmlReader.NodeType == XmlNodeType.EndElement)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        identifierCheck = false;
                    }
                }

                //in subquestion types 7 and 8 it's impossible to read their identifiers before reading (at least a part) of their text
                if (subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer
                    || subquestionType == EnumTypes.SubquestionType.GapMatch)
                {
                    if (xmlReader.Name == "p" && xmlReader.NodeType == XmlNodeType.Element)
                    {
                        pTagVisited = true;
                        if (subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                        {
                            subquestionTextTemporary = xmlReader.ReadString() + "(DOPLŇTE)";
                            oldNodeCount = nodeCount;
                        }
                        else
                        {
                            if (identifierCheck)
                            {
                                subquestionText += xmlReader.ReadString();
                            }
                        }
                    }

                    //the identifier that has been read matches the identifier of the selected subquestion 
                    if ((subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer) && nodeCount == oldNodeCount && xmlReader.GetAttribute("responseIdentifier") == subquestionIdentifier)
                    {
                        identifierCheck = true;
                    }

                    if (identifierCheck)
                    {
                        if (pTagVisited)
                        {
                            if (subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                            {
                                subquestionText = subquestionTextTemporary;

                                //there may be more text following the gap - this text is added to current subquestion text
                                subquestionText += xmlReader.ReadString();
                            }
                            else
                            {
                                amountOfGaps++;
                                subquestionText += xmlReader.ReadString();
                                if (amountOfGaps <= correctAnswersCount)
                                {
                                    subquestionText += "(DOPLŇTE[" + amountOfGaps + "])";
                                }
                            }
                        }

                        //stops adding text to subquestion text after </p> tag is reached
                        if (pTagVisited && xmlReader.Name == "p")
                        {
                            pTagVisited = false;
                        }
                    }
                }
                else
                {
                    if (xmlReader.Name == "prompt")
                    {
                        //using ReadSubtree ensures that every part of subquestion text is properly read and added
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            while (innerReader.Read())
                            {
                                subquestionText += innerReader.ReadString();
                            }
                        }
                    }

                    //in this type of subquestion it is necessary to add text located in some of the simpleAssociableChoice tags to the subquestion text
                    if (subquestionType == EnumTypes.SubquestionType.MultipleQuestions)
                    {
                        if (xmlReader.Name == "simpleMatchSet")
                        {
                            simpleMatchSetCounter++;
                        }

                        if (simpleMatchSetCounter == 2 && xmlReader.Name == "simpleMatchSet")
                        {
                            subquestionText += "(";
                        }

                        if (simpleMatchSetCounter == 3)
                        {
                            if (xmlReader.Name == "simpleAssociableChoice")
                            {
                                subquestionText += xmlReader.ReadString() + ", ";
                            }
                        }

                        if (simpleMatchSetCounter == 4 && xmlReader.Name == "simpleMatchSet")
                        {
                            subquestionText = subquestionText.Substring(0, subquestionText.Length - 2);//remove reduntant comma
                            subquestionText += ")";
                        }
                    }
                }
            }

            //subquestion text may still be empty due to the text being located in <div> tag - correction
            if (subquestionText == "")
            {
                nodeCount = 0;
                identifierCheck = false;
                xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
                while (xmlReader.Read())
                {
                    nodeCount++;
                    if (xmlReader.Name == "div" && xmlReader.GetAttribute("class") == "col-12" && xmlReader.NodeType == XmlNodeType.Element)
                    {
                        divTagVisited = true;
                        if (subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                        {
                            subquestionTextTemporary = xmlReader.ReadString() + "(DOPLŇTE)";
                            oldNodeCount = nodeCount;
                        }
                        else
                        {
                            if (identifierCheck)
                            {
                                subquestionText += xmlReader.ReadString();
                            }
                        }
                    }

                    if ((subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                        && nodeCount == oldNodeCount && xmlReader.GetAttribute("responseIdentifier") == subquestionIdentifier)
                    {
                        identifierCheck = true;
                    }

                    if (identifierCheck)
                    {
                        if (divTagVisited)
                        {
                            if (subquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill || subquestionType == EnumTypes.SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                            {
                                subquestionText = subquestionTextTemporary;
                                subquestionText += xmlReader.ReadString();
                            }
                            else
                            {
                                amountOfGaps++;
                                subquestionText += xmlReader.ReadString();
                                if (amountOfGaps <= correctAnswersCount)
                                {
                                    subquestionText += "(DOPLŇTE[" + amountOfGaps + "])";
                                }
                            }
                        }

                        if (divTagVisited && xmlReader.Name == "div" && xmlReader.NodeType == XmlNodeType.EndElement)
                        {
                            break;//after the correct subquestion text has been read it's necessary to exit the loop to prevent reading further text that may not belong to this subquestion

                        }
                    }
                }
            }
            return subquestionText;
        }

        /// <summary>
        /// Returns the subquestion image source (if one exists)
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the subquestion image source</returns>
        public string GetSubquestionImage(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            string imageSource = "";

            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                    }
                }

                if (xmlReader.Name == "img")
                {
                    imageSource = "/images/" + testNameIdentifier + "/items/" + questionNumberIdentifier + "/" + xmlReader.GetAttribute("src");
                    return imageSource;
                }
            }

            return imageSource;
        }

        /// <summary>
        /// Returns the list of possible answers of the selected subquestion
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the list of possible answers</returns>
        public string[] GetPossibleAnswerList(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, EnumTypes.SubquestionType subquestionType)
        {
            List<string> possibleAnswerList = new List<string>();
            int simpleMatchSetCounter = 0;//for subquestionType 4

            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                var name = xmlReader.Name;
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                        else
                        {
                            if (name == "sliderInteraction")
                            {
                                string lowerBound = "";
                                string upperBound = "";
                                if (xmlReader.GetAttribute("lowerBound") != null)
                                {
                                    lowerBound = xmlReader.GetAttribute("lowerBound")!;
                                }
                                if (xmlReader.GetAttribute("upperBound") != null)
                                {
                                    upperBound = xmlReader.GetAttribute("upperBound")!;
                                }
                                possibleAnswerList.Add(lowerBound + " - " + upperBound);
                            }
                        }
                    }
                }

                if (subquestionType != EnumTypes.SubquestionType.MultipleQuestions)
                {
                    if (name == "simpleAssociableChoice" || name == "gapText" || name == "simpleChoice" || name == "inlineChoice")
                    {
                        string possibleAnswer = xmlReader.ReadString();
                        possibleAnswerList.Add(possibleAnswer);
                    }
                }
                else
                {
                    //in case the subquestion type is 4, not every option located within a simpleAssociableChoice can be added (some have already been added to the subquestion text)
                    if (name == "simpleMatchSet")
                    {
                        simpleMatchSetCounter++;
                    }

                    if (name == "simpleAssociableChoice" && simpleMatchSetCounter == 1)
                    {
                        string possibleAnswer = xmlReader.ReadString();
                        possibleAnswerList.Add(possibleAnswer);
                    }
                }
            }

            return possibleAnswerList.ToArray();
        }

        /// <summary>
        /// Returns the list of answer identifiers (tuples that contain the identifier of the answer and the text of the answer)
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the list of answer identifiers</returns>
        public List<(string, string)> GetAnswerIdentifiers(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            List<(string, string)> answerIdentifiers = new List<(string, string)>();

            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                var name = xmlReader.Name;
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != selectedSubquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                    }

                    if (name == "simpleChoice" || name == "simpleAssociableChoice" || name == "gapText" || name == "inlineChoice")
                    {
                        string answerIdentifier = "";
                        if (xmlReader.GetAttribute("identifier") != null)
                        {
                            answerIdentifier = xmlReader.GetAttribute("identifier")!;
                        }
                        string answerText = xmlReader.ReadString();
                        answerIdentifiers.Add((answerIdentifier, answerText));
                    }
                }
            }
            return answerIdentifiers;
        }

        /// <summary>
        /// Returns the list of correct answers of the selected subquestion
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the list of correct answers</returns>
        public string[] GetCorrectAnswerList(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, EnumTypes.SubquestionType subquestionType)
        {
            List<string> correctIdentifierList = new List<string>();//identifiers of correct choices
            List<string> correctAnswerList = new List<string>();//text of correct choices
            List<(string, string)> answerIdentifiers = GetAnswerIdentifiers(selectedSubquestionIdentifier, testNameIdentifier, questionNumberIdentifier);

            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    string subquestionIdentifier = "";
                    if (xmlReader.GetAttribute("identifier") != null)
                    {
                        subquestionIdentifier = xmlReader.GetAttribute("identifier")!;
                    }
                    if (subquestionIdentifier != selectedSubquestionIdentifier)
                    {
                        xmlReader.Skip();
                    }
                }
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.Name == "value")
                {
                    correctIdentifierList.Add(xmlReader.ReadString());
                }

                //skips the <value> tag that's not relevant to correct answers
                if (xmlReader.Name == "outcomeDeclaration")
                {
                    xmlReader.Skip();
                }
            }

            //slider - the correct answer is one number
            if (subquestionType == EnumTypes.SubquestionType.Slider)
            {
                correctAnswerList = correctIdentifierList;
            }

            for (int i = 0; i < correctIdentifierList.Count; i++)
            {
                for (int j = 0; j < answerIdentifiers.Count; j++)
                {
                    //correct answers of these subquestion types consist of doubles (such as matching two terms)
                    if (subquestionType == EnumTypes.SubquestionType.MatchingElements || subquestionType == EnumTypes.SubquestionType.MultipleQuestions)
                    {
                        string[] splitIdentifiersBySpace = correctIdentifierList[i].Split(" ");
                        if (splitIdentifiersBySpace[0] == answerIdentifiers[j].Item1)
                        {
                            if (correctAnswerList.Count <= i)
                            {
                                correctAnswerList.Add(answerIdentifiers[j].Item2);
                            }
                            else
                            {
                                correctAnswerList[i] = answerIdentifiers[j].Item2 + " -> " + correctAnswerList[i];
                            }
                        }
                        if (splitIdentifiersBySpace[1] == answerIdentifiers[j].Item1)
                        {
                            if (correctAnswerList.Count <= i)
                            {
                                correctAnswerList.Add(answerIdentifiers[j].Item2);
                            }
                            else
                            {
                                correctAnswerList[i] = answerIdentifiers[j].Item2 + " -> " + correctAnswerList[i];
                            }
                        }
                    }
                    //correct answers of this type of subquestion are entered into gaps
                    else if (subquestionType == EnumTypes.SubquestionType.GapMatch)
                    {
                        string[] splitIdentifiersBySpace = correctIdentifierList[i].Split(" ");
                        if (splitIdentifiersBySpace[0] == answerIdentifiers[j].Item1)
                        {
                            correctAnswerList.Add("[" + (correctAnswerList.Count + 1) + "] - " + answerIdentifiers[j].Item2);
                        }
                    }
                    else
                    {
                        if (correctIdentifierList[i] == answerIdentifiers[j].Item1)
                        {
                            correctAnswerList.Add(answerIdentifiers[j].Item2);
                        }
                    }
                }
            }
            return correctAnswerList.ToArray();
        }
    }
}