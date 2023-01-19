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
using System.Threading.Tasks;

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

        public async Task<List<TestTemplate>> GetTestTemplates(string login)
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.Owner)
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplates)
                .ThenInclude(t => t.SubquestionTemplates)
                .Where(t => t.OwnerLogin == login).ToListAsync();
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

        public async Task<List<Subject>> GetSubjects()
        {
            return await GetSubjectDbSet().Include(s => s.Guarantor).ToListAsync();
        }

        public async Task<Subject?> GetSubjectById(int subjectId)
        {
            return await GetSubjectDbSet().Include(s => s.Students).FirstOrDefaultAsync(s => s.SubjectId == subjectId);
        }

        public async Task<string> AddSubject(Subject subject)
        {
            return await dataFunctions.AddSubject(subject);
        }

        public async Task<string> EditSubject(Subject subject, User user)
        {
            Subject? oldSubject = await GetSubjectById(subject.SubjectId);
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
                    oldSubject.Students = subject.Students;

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

        public bool CanUserModifyTemplate(string currentUserlogin, string templateOwnerLogin)
        {
            if(currentUserlogin == templateOwnerLogin)
            {
                return true;
            }
            return false;
        }

        public async Task<string> AddTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            (string? errorMessage, testTemplate) = await ValidateTestTemplate(testTemplate, subjectId);
            if(errorMessage != null)
            {
                return errorMessage;
            }
            else
            {
                return await dataFunctions.AddTestTemplate(testTemplate);
            }
        }

        public async Task<string> EditTestTemplate(string login, TestTemplate testTemplate, string subjectId)
        {
            if(!CanUserModifyTemplate(login, testTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            (string? errorMessage, testTemplate) = await ValidateTestTemplate(testTemplate, subjectId);
            if (errorMessage != null)
            {
                return errorMessage;
            }
            else
            {
                return await dataFunctions.EditTestTemplate(testTemplate);
            }
        }

        public async Task<(string?, TestTemplate)> ValidateTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string? errorMessage = null;
            if (subjectId == "")
            {
                errorMessage = "Chyba: nevyplněný předmět.";
            }
            else
            {
                Subject? subject = await GetSubjectById(int.Parse(subjectId));
                if (subject == null)
                {
                    errorMessage = "Chyba: předmět nenalezen.";
                }
                else
                {
                    testTemplate.Subject = subject;
                }
            }

            if(testTemplate.Title.Length == 0)
            {
                errorMessage = "Chyba: nevyplněný nadpis.";
            }

            if(testTemplate.NegativePoints != NegativePoints.Enabled && testTemplate.NegativePoints != NegativePoints.EnabledForQuestion
                && testTemplate.NegativePoints != NegativePoints.Disabled)
            {
                errorMessage = "Chyba: nekompletní zadání otázky (záporné body).";
            }

            if(testTemplate.MinimumPoints != 0)//new test templates always have minimum points value of 0
            {
                if (testTemplate.MinimumPoints > GetTestTemplatePointsSum(await GetTestTemplate(testTemplate.TestTemplateId)))
                {
                    errorMessage = "Chyba: minimální možný počet bodů pro tento test nemůže být vyšší než " + GetTestTemplatePointsSum(testTemplate);
                }
                else if (testTemplate.MinimumPoints < 0)
                {
                    errorMessage = "Chyba: minimální možný počet bodů pro tento test nemůže být nižší než 0.";
                }
            }
            
            if (testTemplate.StartDate < DateTime.Now)
            {
                errorMessage = "Chyba: čas začátku testu nemůže být v minulosti.";
            }
            if (testTemplate.EndDate < DateTime.Now)
            {
                errorMessage = "Chyba: čas začátku testu nemůže být v minulosti.";
            }
            if(testTemplate.EndDate < testTemplate.StartDate)
            {
                errorMessage = "Chyba: čas konce testu nemůže být přes časem začátku testu.";
            }

            return (errorMessage, testTemplate);
        }

        public bool CanUserEditTestTemplate(TestTemplate testTemplate, string login)
        {
            if (testTemplate.IsTestingData)
            {
                return true;
            }
            bool startTimeCheck = false;
            bool userLoginCheck = false;
            if(testTemplate.StartDate > DateTime.Now)
            {
                startTimeCheck = true;
            }
            if(login == testTemplate.OwnerLogin)
            {
                userLoginCheck = true;
            }
            if(!startTimeCheck || !userLoginCheck)
            {
                return false;
            }
            return true;
        }

        public async Task<string> DeleteTestTemplates(string login)
        {
            return await dataFunctions.DeleteTestTemplates(login);
        }

        public async Task<string> DeleteTestTemplate(string login, int testTemplateId, string webRootPath)
        {
            TestTemplate testTemplate = GetTestTemplateDbSet().First(t => t.OwnerLogin == login && t.TestTemplateId == testTemplateId);
            if (!CanUserModifyTemplate(login, testTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.DeleteTestTemplate(testTemplate, webRootPath);
        }

        public async Task<List<QuestionTemplate>> GetQuestionTemplates(int testTemplateId)
        {
             return await GetQuestionTemplateDbSet()
                 .Include(q => q.SubquestionTemplates)
                 .Include(q => q.TestTemplate)
                 .ThenInclude(q => q.Subject)
                 .Where(q => q.TestTemplate.TestTemplateId == testTemplateId).ToListAsync();
        }

        public async Task<QuestionTemplate> GetQuestionTemplate(int questionTemplateId)
        {
            return await dataFunctions.GetQuestionTemplate(questionTemplateId);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string? errorMessage = ValidateQuestionTemplate(questionTemplate);
            if(errorMessage != null)
            {
                return errorMessage;
            }
            else
            {
                questionTemplate.SubquestionTemplates = new List<SubquestionTemplate>();
                return await dataFunctions.AddQuestionTemplate(questionTemplate);
            }
        }

        public string? ValidateQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string? errorMessage = null;
            if(questionTemplate.Title.Length == 0)
            {
                errorMessage = "Chyba: nevyplněný nadpis.";
            }
            return errorMessage;
        }

        public async Task<string> EditQuestionTemplate(QuestionTemplate questionTemplate, string login)
        {
            if (!CanUserModifyTemplate(login, questionTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.EditQuestionTemplate(questionTemplate);
        }

        public async Task<string> DeleteQuestionTemplate(string login, int questionTemplateId, string webRootPath)
        {
            QuestionTemplate questionTemplate = await GetQuestionTemplate(questionTemplateId);
            if (!CanUserModifyTemplate(login, questionTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.DeleteQuestionTemplate(questionTemplateId, webRootPath);
        }

        public async Task<List<SubquestionTemplate>> GetSubquestionTemplates(int questionTemplateId)
        {
            return await GetSubquestionTemplateDbSet()
                .Include(s => s.QuestionTemplate)
                .Include(s => s.QuestionTemplate.TestTemplate)
                .Where(s => s.QuestionTemplateId == questionTemplateId).ToListAsync();
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            return await dataFunctions.AddSubquestionTemplate(subquestionTemplate, image, webRootPath);
        }

        public async Task<string> EditSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath, string login)
        {
            string message;
            try
            {
                SubquestionTemplate oldSubquestionTemplate = await GetSubquestionTemplate(subquestionTemplate.SubquestionTemplateId);
                if (!CanUserModifyTemplate(login, oldSubquestionTemplate.OwnerLogin))
                {
                    return "K této akci nemáte oprávnění.";
                }
                oldSubquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText;
                oldSubquestionTemplate.PossibleAnswers = subquestionTemplate.PossibleAnswers;
                oldSubquestionTemplate.CorrectAnswers = subquestionTemplate.CorrectAnswers;
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
                    subquestionTextArray[i] = subquestionTextArray[i].Replace("|", "");//replace gap separator
                    subquestionTextArray[i] = subquestionTextArray[i].Replace(";", "");//replace answer separator
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
                subquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText.Replace("|", "");//replace gap separator
                subquestionTemplate.SubquestionText = subquestionTemplate.SubquestionText.Replace(";", "");//replace answer separator
            }

            //change possible and correct answer lists if necessary
            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType.MatchingElements:
                    string[] correctAnswerList = new string[(subquestionTemplate.CorrectAnswers.Length) / 2];
                    for(int i = 0; i < subquestionTemplate.CorrectAnswers.Length; i++)
                    {
                        if(i % 2 == 1)
                        {
                            continue;
                        }
                        else
                        {
                            int index = i / 2;
                            correctAnswerList[index] = subquestionTemplate.CorrectAnswers[i] + "|" + subquestionTemplate.CorrectAnswers[i + 1];
                        }
                    }
                    subquestionTemplate.CorrectAnswers = correctAnswerList;
                    break;
                case SubquestionType.FreeAnswer:
                    subquestionTemplate.PossibleAnswers = new string[0];
                    subquestionTemplate.CorrectAnswers = new string[0];
                    break;
                case SubquestionType.FreeAnswerWithDeterminedCorrectAnswer:
                    subquestionTemplate.PossibleAnswers = new string[0];
                    break;
                case SubquestionType.GapMatch:
                    subquestionTemplate.PossibleAnswers = new string[0];
                    break;
                case SubquestionType.Slider:
                    string[] sliderValuesSplit = sliderValues.Split(",");
                    subquestionTemplate.PossibleAnswers = new string[] { sliderValuesSplit[0], sliderValuesSplit[1] };
                    subquestionTemplate.CorrectAnswers = new string[] { sliderValuesSplit[2] };
                    break;
            }

            //set choice points
            if(subquestionTemplate.SubquestionPoints > 0)
            {
                subquestionTemplate.SubquestionPoints = Math.Round(subquestionTemplate.SubquestionPoints, 2);
                double correctChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(subquestionTemplate.SubquestionPoints,
                    subquestionTemplate.CorrectAnswers, subquestionTemplate.SubquestionType);
                subquestionTemplate.CorrectChoicePoints = correctChoicePoints;
                subquestionTemplate.DefaultWrongChoicePoints = correctChoicePoints * (-1);
                if (subquestionTemplate.WrongChoicePoints != subquestionTemplate.DefaultWrongChoicePoints)//user manually set different wrong choice points than default
                {
                    if (subquestionTemplate.WrongChoicePoints < subquestionTemplate.DefaultWrongChoicePoints)//invalid value - cannot be lesser than default
                    {
                        subquestionTemplate.WrongChoicePoints = subquestionTemplate.DefaultWrongChoicePoints;
                        errorMessage = "Chyba: nejmenší možný počet bodů za špatnou volbu je " + subquestionTemplate.DefaultWrongChoicePoints + ".";
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

            for(int i = 0; i < subquestionTemplate.PossibleAnswers.Length; i++)
            {
                if (subquestionTemplate.PossibleAnswers[i].Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (možná odpověď).";
                }
            }

            for (int i = 0; i < subquestionTemplate.CorrectAnswers.Length; i++)
            {
                if (subquestionTemplate.CorrectAnswers[i].Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (správná odpověď).";
                }
            }

            if (subquestionTemplate.PossibleAnswers.Distinct().Count() != subquestionTemplate.PossibleAnswers.Length)
            {
                errorMessage = "Chyba: duplikátní možná odpověď.";
            }

            if (subquestionTemplate.CorrectAnswers.Distinct().Count() != subquestionTemplate.CorrectAnswers.Length &&
                subquestionTemplate.SubquestionType != SubquestionType.MultipleQuestions)
            {
                errorMessage = "Chyba: duplikátní správná odpověď.";
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers
                 || subquestionTemplate.SubquestionType == SubquestionType.MatchingElements || subquestionTemplate.SubquestionType == SubquestionType.MultipleQuestions
                  || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill
                   || subquestionTemplate.SubquestionType == SubquestionType.Slider)
            {
                if (subquestionTemplate.PossibleAnswers.Length == 0)
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
                if (subquestionTemplate.CorrectAnswers.Length == 0)
                {
                    errorMessage = "Chyba: nekompletní zadání podotázky (správné odpovědi).";
                }
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers
                 || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill)
            {
                for(int i = 0; i < subquestionTemplate.CorrectAnswers.Length; i++)
                {
                    if (!subquestionTemplate.PossibleAnswers.Contains(subquestionTemplate.CorrectAnswers[i]))
                    {
                        errorMessage = "Chyba: nekompletní zadání podotázky (možné a správné odpovědi).";
                    }
                }
            }

            if (subquestionTemplate.SubquestionType == SubquestionType.OrderingElements)
            {
                if(subquestionTemplate.PossibleAnswers.Length != subquestionTemplate.CorrectAnswers.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceMultipleCorrectAnswers)
            {
                if (subquestionTemplate.PossibleAnswers.Length < subquestionTemplate.CorrectAnswers.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MatchingElements)
            {
                if (subquestionTemplate.PossibleAnswers.Length / 2 < subquestionTemplate.CorrectAnswers.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultipleQuestions)
            {
                if (subquestionTemplate.PossibleAnswers.Length != subquestionTemplate.CorrectAnswers.Length)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.FreeAnswer)
            {
                if (subquestionTemplate.PossibleAnswers.Length > 0 || subquestionTemplate.CorrectAnswers.Length > 0)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceSingleCorrectAnswer)
            {
                if (subquestionTemplate.PossibleAnswers.Length <= 1 || subquestionTemplate.CorrectAnswers.Length > 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.MultiChoiceTextFill)
            {
                if (subquestionTemplate.PossibleAnswers.Length <= 1 || subquestionTemplate.CorrectAnswers.Length > 1)
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
                if (subquestionTemplate.PossibleAnswers.Length > 0 || subquestionTemplate.CorrectAnswers.Length > 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.GapMatch)
            {
                string[] subquestionTextGapSplit = subquestionTemplate.SubquestionText.Split("|");
                if (subquestionTemplate.PossibleAnswers.Length > 0 || subquestionTemplate.CorrectAnswers.Length != subquestionTextGapSplit.Length - 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
            }
            else if (subquestionTemplate.SubquestionType == SubquestionType.Slider)
            {
                if (subquestionTemplate.PossibleAnswers.Length != 2 || subquestionTemplate.CorrectAnswers.Length != 1)
                {
                    errorMessage = "Chyba: neplatné zadání podotázky (počet možných/správých odpovědí).";
                }
                int lowerBound = int.Parse(subquestionTemplate.PossibleAnswers[0]);
                int upperBound = int.Parse(subquestionTemplate.PossibleAnswers[1]);
                int answer = int.Parse(subquestionTemplate.CorrectAnswers[0]);
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
        public List<SubquestionTemplate> ProcessSubquestionTemplatesForView(List<SubquestionTemplate> subquestionTemplates)
        {
            List<SubquestionTemplate> processedSubquestionTemplates = new List<SubquestionTemplate>();

            foreach (SubquestionTemplate subquestionTemplate in subquestionTemplates)
            {
                processedSubquestionTemplates.Add(ProcessSubquestionTemplateForView(subquestionTemplate));
            }

            return processedSubquestionTemplates;
        }

        public SubquestionTemplate ProcessSubquestionTemplateForView(SubquestionTemplate subquestionTemplate)
        {
            string[] possibleAnswerList;
            string[] correctAnswerList;

            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType.MatchingElements:
                    correctAnswerList = new string[subquestionTemplate.CorrectAnswers.Length];
                    for (int i = 0; i < subquestionTemplate.CorrectAnswers.Length; i++)
                    {
                        correctAnswerList[i] = subquestionTemplate.CorrectAnswers[i].Replace("|", " -> ");
                    }
                    subquestionTemplate.CorrectAnswers = correctAnswerList;
                    break;
                case SubquestionType.MultipleQuestions:
                    possibleAnswerList = new string[2] { "Ano", "Ne" };
                    correctAnswerList = new string[subquestionTemplate.PossibleAnswers.Length];
                    for (int i = 0; i < subquestionTemplate.PossibleAnswers.Length; i++)
                    {
                        string answer = "";
                        if (subquestionTemplate.CorrectAnswers[i] == "1")
                        {
                            answer = "Ano";
                        }
                        else if (subquestionTemplate.CorrectAnswers[i] == "0")
                        {
                            answer = "Ne";
                        }
                        correctAnswerList[i] = subquestionTemplate.PossibleAnswers[i] + " -> " + answer;
                    }
                    subquestionTemplate.PossibleAnswers = possibleAnswerList;
                    subquestionTemplate.CorrectAnswers = correctAnswerList;
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

                    subquestionTemplate.PossibleAnswers = subquestionTemplate.CorrectAnswers;
                    correctAnswerList = new string[subquestionTemplate.CorrectAnswers.Length];
                    for (int i = 0; i < subquestionTemplate.CorrectAnswers.Length; i++)
                    {
                        correctAnswerList[i] = "[" + (i + 1) + "] - " + subquestionTemplate.CorrectAnswers[i];
                    }
                    subquestionTemplate.CorrectAnswers = correctAnswerList;
                    break;
                case SubquestionType.Slider:
                    possibleAnswerList = new string[] { subquestionTemplate.PossibleAnswers[0] + " - " + subquestionTemplate.PossibleAnswers[1] };
                    subquestionTemplate.PossibleAnswers = possibleAnswerList;
                    break;
            }
            return subquestionTemplate;
        }

        public async Task<string> DeleteSubquestionTemplate(string login, int subquestionTemplateId, string webRootPath)
        {
            SubquestionTemplate subquestionTemplate = await GetSubquestionTemplate(subquestionTemplateId);
            if(!CanUserModifyTemplate(login, subquestionTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.DeleteSubquestionTemplate(subquestionTemplateId, webRootPath);
        }

        public async Task<TestTemplate> GetTestTemplate(int testTemplateId)
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.Owner)
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .FirstAsync(t => t.TestTemplateId == testTemplateId);
        }

        public async Task<SubquestionTemplate> GetSubquestionTemplate(int subquestionTemplateId)
        {
            return await GetSubquestionTemplateDbSet()
                .Include(s => s.QuestionTemplate)
                .Include(s => s.QuestionTemplate.TestTemplate)
                .FirstAsync(s => s.SubquestionTemplateId == subquestionTemplateId);
        }

        public async Task<SubquestionTemplateStatistics> GetSubquestionTemplateStatistics(string login)
        {
            SubquestionTemplateStatistics? subquestionTemplateStatistics = await dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
            if(subquestionTemplateStatistics == null)
            {
                throw Exceptions.SubquestionTemplateStatisticsNotFoundException(login);
            }
            return subquestionTemplateStatistics;
        }

        public async Task<SubquestionTemplateStatistics?> GetSubquestionTemplateStatisticsNullable(string login)
        {
            return await dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(SubquestionTemplate subquestionTemplate, bool subquestionTemplateExists)
        {
            if (subquestionTemplateExists)
            {
                subquestionTemplate = await GetSubquestionTemplate(subquestionTemplate.SubquestionTemplateId);
            }

            string login = subquestionTemplate.OwnerLogin;
            User owner = await dataFunctions.GetUserByLogin(login);

            //check if enough subquestion templates have been added to warrant new model training
            bool retrainModel = false;
            SubquestionTemplateStatistics subquestionTemplateStatistics = await GetSubquestionTemplateStatistics(subquestionTemplate.OwnerLogin);
            int subquestionTemplatesAdded = subquestionTemplateStatistics.SubquestionTemplatesAddedCount;
            if (subquestionTemplatesAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionTemplateModel(owner);
            }

            var testTemplates = await dataFunctions.GetTestTemplateList(owner.Login);
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageTemplatePoints(testTemplates);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageTemplatePoints(testTemplates);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionTemplateRecord currentSubquestionTemplateRecord = DataGenerator.CreateSubquestionTemplateRecord(subquestionTemplate, owner, subjectAveragePointsTuple,
                subquestionTypeAveragePoints, minimumPointsShare);
            SubquestionTemplateStatistics currentSubquestionTemplateStatistics = await GetSubquestionTemplateStatistics(login);
            if (!currentSubquestionTemplateStatistics.EnoughSubquestionTemplatesAdded)
            {
                return "Pro použití této funkce je nutné přidat alespoň 100 zadání podotázek.";
            }
            Model usedModel = currentSubquestionTemplateStatistics.UsedModel;
            double suggestedSubquestionPoints = PythonFunctions.GetSubquestionTemplateSuggestedPoints(login, retrainModel, currentSubquestionTemplateRecord, usedModel);
            if (subquestionTemplatesAdded >= 100)
            {
                subquestionTemplateStatistics = await GetSubquestionTemplateStatistics(login);
                subquestionTemplateStatistics.EnoughSubquestionTemplatesAdded = true;
                subquestionTemplateStatistics.SubquestionTemplatesAddedCount = 0;
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

            return suggestedSubquestionPoints.ToString();
        }

        public async Task RetrainSubquestionTemplateModel(User owner)
        {
            string login = owner.Login;
            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testTemplates = await dataFunctions.GetTestTemplateList(login);
            var subquestionTemplateRecords = DataGenerator.CreateSubquestionTemplateRecords(testTemplates);

            await dataFunctions.SaveSubquestionTemplateRecords(subquestionTemplateRecords, owner);
        }

        public async Task<List<TestTemplate>> GetTestingDataTestTemplates()
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Where(t => t.IsTestingData).ToListAsync();
        }

        public async Task<int> GetTestingDataSubquestionTemplatesCount()
        {
            int testingDataSubquestionTemplates = 0;
            var testTemplates = await GetTestingDataTestTemplates();
            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];
                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);
                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
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

            if(Convert.ToInt32(amountOfSubquestionTemplates) < 1)
            {
                return "Chyba: je nutné zadat kladnou hodnotu počtu testovacích dat.";
            }

            await TestingUsersCheck();
            User owner = await dataFunctions.GetUserByLogin("login");
            var existingTestTemplates = await GetTestingDataTestTemplates();
            if (existingTestTemplates.Count == 0)//no templates exist - we have to add subjects
            {
                await CreateSubjectTestingData(owner);
            }
            List<Subject> testingDataSubjects = await dataFunctions.GetTestingDataSubjects();
            List<TestTemplate> testTemplates = new List<TestTemplate>();

            if (action == "addSubquestionTemplateRandomData")
            {
                testTemplates = DataGenerator.GenerateRandomTestTemplates(existingTestTemplates, Convert.ToInt32(amountOfSubquestionTemplates), testingDataSubjects);
            }
            else if (action == "addSubquestionTemplateCorrelationalData")
            {
                testTemplates = DataGenerator.GenerateCorrelationalTestTemplates(existingTestTemplates, Convert.ToInt32(amountOfSubquestionTemplates), testingDataSubjects);
            }
            message = await dataFunctions.AddTestTemplates(testTemplates, owner);
            string login = "login";
            owner = await dataFunctions.GetUserByLoginAsNoTracking();

            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionTemplateRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testTemplatesToRecord = await dataFunctions.GetTestTemplateList(login);

            var subquestionTemplateRecords = DataGenerator.CreateSubquestionTemplateRecords(testTemplatesToRecord);
            await dataFunctions.SaveSubquestionTemplateRecords(subquestionTemplateRecords, owner);

            dataFunctions.ClearChargeTracker();
            owner = await dataFunctions.GetUserByLoginAsNoTracking();
            var subquestionTemplateStatistics = await GetSubquestionTemplateStatistics(owner.Login);
            if (subquestionTemplateStatistics == null)
            {
                subquestionTemplateStatistics = new SubquestionTemplateStatistics();
                subquestionTemplateStatistics.User = owner;
                subquestionTemplateStatistics.UserLogin = owner.Login;
                subquestionTemplateStatistics.EnoughSubquestionTemplatesAdded = true;
                subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "TemplateNeuralNetwork.py");
                subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "TemplateMachineLearning.py");
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

        public async Task CreateSubjectTestingData(User owner)
        {
            Subject[] testingDataSubjects = new Subject[] { 
                new Subject("Ch", "Chemie", owner, owner.Login, new List<Student>(), true),
                new Subject("Z", "Zeměpis", owner, owner.Login, new List<Student>(), true),
                new Subject("M", "Matematika", owner, owner.Login, new List<Student>(), true),
                new Subject("D", "Dějepis", owner, owner.Login, new List<Student>(), true),
                new Subject("I", "Informatika", owner, owner.Login, new List<Student>(), true) };
            for(int i = 0; i < testingDataSubjects.Length; i++)
            {
                await dataFunctions.AddSubject(testingDataSubjects[i]);
            }
        }

        public async Task TestingUsersCheck()
        {
            User? owner = await dataFunctions.GetUserByLoginNullable("login");
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
            dataFunctions.ExecuteSqlRaw("delete from Subject where IsTestingData = 1");

            //since results are directly linked to templates, they are deleted as well
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where OwnerLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultStatistics where UserLogin = 'login'");

            //delete all models (if they exist)
            string[] testingDataModels = new string[] {
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\templates\\login_NN.pt",
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\templates\\login_LR.sav",
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\results\\login_NN.pt",
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\results\\login_LR.sav"};
            for(int i = 0; i < testingDataModels.Length; i++)
            {
                if (File.Exists(testingDataModels[i]))
                {
                    File.Delete(testingDataModels[i]);
                }
            }

            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> GetTestDifficultyPrediction(string login, int testTemplateId)
        {
            string testDifficultyMessage;
            TestDifficultyStatistics? testDifficultyStatistics = await dataFunctions.GetTestDifficultyStatisticsNullable(login);
            SubquestionResultStatistics? subquestionResultStatistics = await dataFunctions.GetSubquestionResultStatisticsNullable(login);
            //check whether there are enough result statistics to go by
            if (testDifficultyStatistics == null || subquestionResultStatistics == null)
            {
                return "Chyba: nedostatečný počet výsledků testů.;";
            }

            //predicted amount of points that the average student will get for this test
            double testTemplatePredictedPoints = PythonFunctions.GetTestTemplatePredictedPoints(false, login, subquestionResultStatistics.UsedModel, testTemplateId);

            List<TestResult> testResults = await dataFunctions.GetTestResultsByLogin(login);
            List<(int, int, double)> testResultsPointsShare = new List<(int, int, double)>();
            
            //iterate through every test result to obtain the share between student's points and test total points
            for(int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];
                TestTemplate testTemplate = testResult.TestTemplate;

                //skip the test that's currently being predicted as we want it to compare to other tests
                if(testTemplate.TestTemplateId == testTemplateId)
                {
                    continue;
                }

                double testTemplatePointsSum = GetTestTemplatePointsSum(testTemplate);
                double studentsPoints = 0;

                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);

                    for(int k = 0; k < questionResult.SubquestionResults.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(k);
                        studentsPoints += subquestionResult.StudentsPoints;
                    }
                }

                bool matchFound = false;
                for(int l = 0; l < testResultsPointsShare.Count; l++)
                {
                    //record of this test template is already in the testResultPointsShare list - we modify it
                    if (testResultsPointsShare[l].Item1 == testTemplate.TestTemplateId)
                    {
                        matchFound = true;
                        testResultsPointsShare[l] = (testResultsPointsShare[l].Item1, testResultsPointsShare[l].Item2 + 1, testResultsPointsShare[l].Item3 + studentsPoints);
                    }
                }

                //record of this test template is not in the testResultPointsShare list - we add a new one
                if (!matchFound)
                {
                    testResultsPointsShare.Add((testTemplate.TestTemplateId, 1, studentsPoints / testTemplatePointsSum));
                }
            }

            double[] testTemplatesPointsShare = new double[testResultsPointsShare.Count];
            for (int i = 0; i < testResultsPointsShare.Count; i++)
            {
                testTemplatesPointsShare[i] = testResultsPointsShare[i].Item3 / testResultsPointsShare[i].Item2;
            }
            TestTemplate currentTestTemplate = await GetTestTemplate(testTemplateId);
            double currentTestTemplatePointsShare = testTemplatePredictedPoints / GetTestTemplatePointsSum(currentTestTemplate);
            //compare the ratio of predicted test points to total test points with the average points of all test templates (as measured by existing test results)
            double difficulty = currentTestTemplatePointsShare / testTemplatesPointsShare.Average();

            testDifficultyMessage = testTemplatePredictedPoints + ";";
            if (difficulty > 1)//easier than the average test
            {
                difficulty *= 100;
                difficulty = Math.Round(difficulty, 0);
                testDifficultyMessage += "Test je o " + (difficulty - 100) + "% lehčí než průměrný test.";
            }
            else//harder than the average test
            {
                difficulty *= 100;
                difficulty = Math.Round(difficulty, 0);
                testDifficultyMessage += "Test je o " + (100 - difficulty) + "% těžší než průměrný test.";
            }

            return testDifficultyMessage;
        }

        public double GetTestTemplatePointsSum(TestTemplate testTemplate)
        {
            double testPoints = 0;
            for (int i = 0; i < testTemplate.QuestionTemplates.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(i);

                for (int j = 0; j < questionTemplate.SubquestionTemplates.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(j);
                    testPoints += subquestionTemplate.SubquestionPoints;
                }
            }

            return Math.Round(testPoints, 2);
        }

        public int GetTestTemplateSubquestionsCount(TestTemplate testTemplate)
        {
            int subquestionsCount = 0;
            for (int i = 0; i < testTemplate.QuestionTemplates.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(i);

                for (int j = 0; j < questionTemplate.SubquestionTemplates.Count; j++)
                {
                    subquestionsCount++;
                }
            }

            return subquestionsCount;
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
    }
}