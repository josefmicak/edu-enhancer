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

        /// <summary>
        /// Returns the list of test templates of the teacher
        /// </summary>
        /// <param name="login">Teacher's login</param>
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

        public async Task<Subject> GetSubjectById(int subjectId)
        {
            Subject? subject = await GetSubjectDbSet().Include(s => s.Students).FirstOrDefaultAsync(s => s.SubjectId == subjectId);
            if(subject == null)
            {
                throw Exceptions.SubjectNotFoundException(subjectId);
            }
            return subject;
        }

        public async Task<Subject?> GetSubjectByIdNullable(int subjectId)
        {
            return await GetSubjectDbSet().Include(s => s.Students).FirstOrDefaultAsync(s => s.SubjectId == subjectId);
        }

        public async Task<string> AddSubject(Subject subject)
        {
            return await dataFunctions.AddSubject(subject);
        }

        /// <summary>
        /// Updates subject
        /// </summary>
        /// <param name="subject">Subject</param>
        /// <param name="user">User performing the update (only subject's guarantor can make changes to the subject)</param>
        public async Task<string> EditSubject(Subject subject, User user)
        {
            Subject oldSubject = await GetSubjectById(subject.SubjectId);
            if(oldSubject.GuarantorLogin != user.Login && user.Role != Role.MainAdmin)
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

        /// <summary>
        /// Valitates subject (checks whether all the fields are properly filled)
        /// </summary>
        public string? ValidateSubject(Subject subject)
        {
            string? errorMessage = null;
            if(subject.Name == null || subject.Abbreviation == null)
            {
                errorMessage = "Chyba: U předmětu schází určité údaje.";
            }

            return errorMessage;
        }

        /// <summary>
        /// Deletes subject
        /// </summary>
        /// <param name="subject">Subject</param>
        /// <param name="user">User performing the deletion (only subject's guarantor can make changes to the subject)</param>
        public async Task<string> DeleteSubject(Subject subject, User user)
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

        /// <summary>
        /// Checks whether (test/question/subquestion) template can be modified by the user (templates can only be modified by its owners)
        /// </summary>
        /// <param name="currentUserlogin">Current user's login</param>
        /// <param name="templateOwnerLogin">Template owner's login</param>
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

        /// <summary>
        /// Updates test template
        /// </summary>
        /// <param name="login">Teacher's login</param>
        /// <param name="testTemplate">Test template (with updated values)</param>
        /// <param name="subjectId">Selected subject Id of updated test template</param>
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

        /// <summary>
        /// Validates whether test template's parameters valid and the test template can be added/edited
        /// This function is used to prevent the student from invalid data  being submitted(which could result in unexpected behavior)
        /// </summary>
        /// <param name="testTemplate">Test template that is yet to be validated</param>
        /// <param name="subjectId">Selected subject Id of updated test template</param>
        public async Task<(string?, TestTemplate)> ValidateTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string? errorMessage = null;
            if (subjectId == "")
            {
                errorMessage = "Chyba: nevyplněný předmět.";
            }
            else
            {
                Subject subject = await GetSubjectById(int.Parse(subjectId));
                testTemplate.Subject = subject;
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

        /// <summary>
        /// Checks whether test template can be modified by the teacher
        /// - test template can only be edited by its owner
        /// - test template can only be edited before test template's start date
        /// </summary>
        /// <param name="testTemplate">Test template</param>
        /// <param name="login">Teacher's login</param>
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

        /// <summary>
        /// Deletes test templates of the teacher
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<string> DeleteTestTemplates(string login)
        {
            return await dataFunctions.DeleteTestTemplates(login);
        }

        /// <summary>
        /// Deletes a test result
        /// </summary>
        /// <param name="login">Teacher's login</param>
        /// <param name="testTemplateId">Id of test template</param>
        /// <param name="webRootPath">Web root path - used to locate images belonging to test template that are deleted together with the test template</param>
        public async Task<string> DeleteTestTemplate(string login, int testTemplateId, string webRootPath)
        {
            TestTemplate testTemplate = GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .First(t => t.OwnerLogin == login && t.TestTemplateId == testTemplateId);
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

        /// <summary>
        /// Validates whether question template is valid and can be added or edited
        /// </summary>
        public string? ValidateQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string? errorMessage = null;
            if(questionTemplate.Title.Length == 0)
            {
                errorMessage = "Chyba: nevyplněný nadpis.";
            }
            return errorMessage;
        }

        /// <summary>
        /// Updates question template
        /// </summary>
        /// <param name="questionTemplate">Question template</param>
        /// <param name="login">Login of the user performing the update (only template's owner can make changes to the template)</param>
        public async Task<string> EditQuestionTemplate(QuestionTemplate questionTemplate, string login)
        {
            if (!CanUserModifyTemplate(login, questionTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.EditQuestionTemplate(questionTemplate);
        }

        /// <summary>
        /// Deletes a question template
        /// </summary>
        /// <param name="login">Teacher's login</param>
        /// <param name="questionTemplateId">Id of question template</param>
        /// <param name="webRootPath">Web root path - used to locate images belonging to test template that are deleted together with the question template</param>
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

        /// <summary>
        /// Updates subquestion template
        /// </summary>
        /// <param name="subquestionTemplate">Subquestion template</param>
        /// <param name="image">Image - can be null</param>
        /// <param name="webRootPath">Web root path - all images are saved to the Uploads folder located within the web root path</param>
        /// <param name="login">Teacher's login</param>
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
        /// Validates whether subquestion template's fields are valid and subquestion template can be added/edited
        /// This function is used to prevent invalid from being added (which could result in unexpected behavior)
        /// </summary>
        /// <param name="subquestionTemplate">Subquestion template that is yet to be validated</param>
        /// <param name="subquestionTextArray">Array of subquestion text fragments (used only for subquestion types that include gaps - 7, 8 and 9)</param>
        /// <param name="subquestionTextArray">String that includes slider values (possible and correct answers; used only for subquestion type slider - 10)</param>
        /// <param name="image">Image - can be null</param>
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
                    if(subquestionTextArray[i] != null)
                    {
                        subquestionTextArray[i] = subquestionTextArray[i].Replace("|", "");//replace gap separator
                        subquestionTextArray[i] = subquestionTextArray[i].Replace(";", "");//replace answer separator
                    }

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
                if(subquestionTemplate.WrongChoicePoints >= 0)
                {
                    errorMessage = "Chyba: za špatnou volbu nemůže být udělen nezáporný počet bodů.";
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
        /// Processes subquestion templates for view - certain chars are replaced with more intuitive strings
        /// <summary>
        public List<SubquestionTemplate> ProcessSubquestionTemplatesForView(List<SubquestionTemplate> subquestionTemplates)
        {
            List<SubquestionTemplate> processedSubquestionTemplates = new List<SubquestionTemplate>();

            foreach (SubquestionTemplate subquestionTemplate in subquestionTemplates)
            {
                processedSubquestionTemplates.Add(ProcessSubquestionTemplateForView(subquestionTemplate));
            }

            return processedSubquestionTemplates;
        }

        /// <summary>
        /// Processes subquestion template for view - certain chars are replaced with more intuitive strings
        /// <summary>
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

        /// <summary>
        /// Deletes a subquestion template
        /// </summary>
        /// <param name="login">Teacher's login</param>
        /// <param name="subquestionTemplateId">Id of subquestion template</param>
        /// <param name="webRootPath">Web root path - used to locate image that belongs to subquestion template</param>
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

        /// <summary>
        /// Returns teacher's subquestion template statistics - throws exception if statistics are not found
        /// </summary>
        public async Task<SubquestionTemplateStatistics> GetSubquestionTemplateStatistics(string login)
        {
            SubquestionTemplateStatistics? subquestionTemplateStatistics = await dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
            if(subquestionTemplateStatistics == null)
            {
                throw Exceptions.SubquestionTemplateStatisticsNotFoundException(login);
            }
            return subquestionTemplateStatistics;
        }

        /// <summary>
        /// Returns teacher's subquestion template statistics - doesn't throw an exception if statistics are not found
        /// </summary>
        public async Task<SubquestionTemplateStatistics?> GetSubquestionTemplateStatisticsNullable(string login)
        {
            return await dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
        }

        /// <summary>
        /// Suggests the amount of points that the subquestion template should get
        /// </summary>
        /// <param name="subquestionTemplate">Subquestion template</param>
        /// <param name="subquestionTemplateExists">Suggestion can be made for both existing and non-existing subquestion templates</param>
        /// <param name="isNUnitTest">Whether function is ran as a part of an NUnit test (false by default)</param>
        public async Task<string> GetSubquestionTemplatePointsSuggestion(SubquestionTemplate subquestionTemplate, bool subquestionTemplateExists, bool isNUnitTest = false)
        {
            string login = subquestionTemplate.OwnerLogin;
            User owner = await dataFunctions.GetUserByLogin(login);

            SubquestionTemplateStatistics currentSubquestionTemplateStatistics = await GetSubquestionTemplateStatistics(login);
            if (!currentSubquestionTemplateStatistics.EnoughSubquestionTemplatesAdded)
            {
                return "Pro použití této funkce je nutné přidat alespoň 100 zadání podotázek.";
            }

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

            double subquestionPoints = subquestionTemplate.SubquestionPoints;
            if (subquestionPoints == 0)//prevent division by 0
            {
                subquestionPoints = 10;
            }
            subquestionTemplate.CorrectChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(
                subquestionPoints, subquestionTemplate.CorrectAnswers, subquestionTemplate.SubquestionType);
            subquestionTemplate.DefaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
            subquestionTemplate.WrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);

            SubquestionTemplateRecord currentSubquestionTemplateRecord = DataGenerator.CreateSubquestionTemplateRecord(subquestionTemplate, owner, subjectAveragePointsTuple,
                subquestionTypeAveragePoints, minimumPointsShare);
            Model usedModel = currentSubquestionTemplateStatistics.UsedModel;
            double suggestedSubquestionPoints = 0;
            if (!isNUnitTest)
            {
                suggestedSubquestionPoints = PythonFunctions.GetSubquestionTemplateSuggestedPoints(login, retrainModel, currentSubquestionTemplateRecord, usedModel);
            }
            if (subquestionTemplatesAdded >= 100)
            {
                subquestionTemplateStatistics = await GetSubquestionTemplateStatistics(login);
                subquestionTemplateStatistics.EnoughSubquestionTemplatesAdded = true;
                subquestionTemplateStatistics.SubquestionTemplatesAddedCount = 0;
                if (!isNUnitTest)
                {
                    subquestionTemplateStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateNeuralNetwork.py");
                    subquestionTemplateStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "TemplateMachineLearning.py");
                }
                else
                {
                    subquestionTemplateStatistics.NeuralNetworkAccuracy = 1;
                    subquestionTemplateStatistics.MachineLearningAccuracy = 1;
                }
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

        /// <summary>
        /// Deletes all existing subquestion template records of the teacher and replaces them with records of current subquestion templates
        /// <param name="owner">Teacher</param>
        /// </summary>
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

        /// <summary>
        /// Returns all test templates marked as testing data
        /// </summary>
        public async Task<List<TestTemplate>> GetTestingDataTestTemplates()
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Where(t => t.IsTestingData).ToListAsync();
        }

        /// <summary>
        /// Returns the amount of subquestion templates marked as testing data
        /// </summary>
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

        /// <summary>
        /// Creates a number of subquestion templates marked as testing data
        /// This function is accessible via the ManageArtificialIntelligence view
        /// </summary>
        /// <param name="action">addSubquestionTemplateRandomData or addSubquestionTemplateCorrelationalData</param>
        /// <param name="amountOfSubquestionTemplates">Amount of subquestion templates to be created</param>
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
            message = await dataFunctions.AddTestTemplates(testTemplates);
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
            SubquestionTemplateStatistics? subquestionTemplateStatistics = await GetSubquestionTemplateStatisticsNullable(owner.Login);
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

        /// <summary>
        /// Creates a number of subjects marked as testing data
        /// </summary>
        /// <param name="owner">Owner - testing data teacher</param>
        public async Task CreateSubjectTestingData(User owner)
        {
            Student student = await dataFunctions.GetStudentByLogin("testingstudent");
            Subject[] testingDataSubjects = new Subject[] { 
                new Subject("Ch", "Chemie", owner, owner.Login, new List<Student> { student }, true),
                new Subject("Z", "Zeměpis", owner, owner.Login, new List<Student> { student }, true),
                new Subject("M", "Matematika", owner, owner.Login, new List<Student> { student }, true),
                new Subject("D", "Dějepis", owner, owner.Login, new List<Student> { student }, true),
                new Subject("I", "Informatika", owner, owner.Login, new List<Student> { student }, true) };
            for(int i = 0; i < testingDataSubjects.Length; i++)
            {
                await dataFunctions.AddSubject(testingDataSubjects[i]);
            }
        }

        /// <summary>
        /// Checks whether users marked as testing data exist, and creates then if they don't
        /// </summary>
        public async Task TestingUsersCheck()
        {
            User? owner = await dataFunctions.GetUserByLoginNullable("login");
            if(owner == null)
            {
                owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
                await dataFunctions.AddUser(owner);
            }

            Student? student = await dataFunctions.GetStudentByLoginNullable("testingstudent");
            if (student == null)
            {
                student = new Student() { Login = "testingstudent", Email = "studentemail", FirstName = "name", LastName = "surname", IsTestingData = true };
                await dataFunctions.AddStudent(student);
            }
        }

        /// <summary>
        /// Deletes all testing data related to templates (and results too as they are dependent on templates)
        /// </summary>
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

        /// <summary>
        /// Predicts the amount of points that students will receive for the test on average
        /// <param name="login">Teacher's login</param>
        /// <param name="testTemplateId">Id of the test template</param>
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

        /// <summary>
        /// Returns the max amount of points that is awarded for the test
        /// </summary>
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

        /// <summary>
        /// Returns the amount of subquestion templates that belong to the test template
        /// </summary>
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

        /// <summary>
        /// Returns the description of subquestion type based on its identifying number
        /// </summary>
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

        /// <summary>
        /// Array containing descriptions of subquestion types
        /// </summary>
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

        /// <summary>
        /// Returns the description of subquestion type based on its identifying number
        /// </summary>
        public string[] GetSubquestionTypeTextArray()
        {
            return SubquestionTypeTextArray;
        }

        /// <summary>
        /// Creates data that is used during NUnit tests - users, students, templates, results, statistics..
        /// </summary>
        public async Task<string> CreateNUnitData()
        {
            //before NUnit data is added, all previously existing records of NUnit data is deleted
            await DeleteNUnitData();

            User admin = new User();
            admin.Login = "nunittestingadmin";
            admin.Email = "EmailAdmin";
            admin.FirstName = "Name";
            admin.LastName = "Surname";
            admin.Role = EnumTypes.Role.Admin;
            admin.IsTestingData = true;
            await dataFunctions.AddUser(admin);

            User teacher = new User();
            teacher.Login = "nunittestingteacher";
            teacher.Email = "EmailTeacher";
            teacher.FirstName = "Name";
            teacher.LastName = "Surname";
            teacher.Role = EnumTypes.Role.Teacher;
            teacher.IsTestingData = true;
            await dataFunctions.AddUser(teacher);

            Student student = new Student();
            student.Login = "nunittestingstudent";
            student.Email = "EmailStudent";
            student.FirstName = "Name";
            student.LastName = "Surname";
            student.IsTestingData = true;
            await dataFunctions.AddStudent(student);

            Subject subject = new Subject();
            subject.Abbreviation = "POS";
            subject.Name = "Počítačové sítě";
            subject.Guarantor = await dataFunctions.GetUserByLogin("nunittestingadmin");
            subject.GuarantorLogin = "nunittestingadmin";
            subject.IsTestingData = true;
            subject.Students = new List<Student> { await dataFunctions.GetStudentByLogin("nunittestingstudent") };
            await dataFunctions.AddSubject(subject);

            TestTemplate testTemplateNUnit = new TestTemplate();
            testTemplateNUnit.Title = "postest";
            testTemplateNUnit.MinimumPoints = 0;
            testTemplateNUnit.StartDate = new DateTime(2022, 12, 30, 00, 00, 00);
            testTemplateNUnit.EndDate = new DateTime(2022, 12, 31, 00, 00, 00);
            testTemplateNUnit.Subject = await GetSubjectDbSet().FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
            testTemplateNUnit.Owner = await dataFunctions.GetUserByLogin("nunittestingadmin");
            testTemplateNUnit.OwnerLogin = "nunittestingadmin";
            testTemplateNUnit.IsTestingData = true;
            testTemplateNUnit.NegativePoints = NegativePoints.Enabled;
            testTemplateNUnit.QuestionTemplates = new List<QuestionTemplate>();
            await dataFunctions.AddTestTemplate(testTemplateNUnit);

            QuestionTemplate questionTemplateNUnit = new QuestionTemplate();
            questionTemplateNUnit.Title = "NUnit title";
            questionTemplateNUnit.OwnerLogin = "nunittestingadmin";
            questionTemplateNUnit.TestTemplate = await GetTestTemplateDbSet().FirstAsync(t => t.Title == "postest" && t.OwnerLogin == "nunittestingadmin");
            questionTemplateNUnit.SubquestionTemplates = new List<SubquestionTemplate>();
            await dataFunctions.AddQuestionTemplate(questionTemplateNUnit);

            SubquestionTemplate subquestionTemplateNUnit = new SubquestionTemplate();
            subquestionTemplateNUnit.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
            subquestionTemplateNUnit.SubquestionText = "NUnit subquestion text";
            subquestionTemplateNUnit.PossibleAnswers = new string[] { "test1", "test2", "test3" };
            subquestionTemplateNUnit.CorrectAnswers = new string[] { "test1", "test2", "test3" };
            subquestionTemplateNUnit.SubquestionPoints = 10;
            subquestionTemplateNUnit.CorrectChoicePoints = 10;
            subquestionTemplateNUnit.DefaultWrongChoicePoints = -10;
            subquestionTemplateNUnit.WrongChoicePoints = -10;
            subquestionTemplateNUnit.QuestionTemplate = await GetQuestionTemplateDbSet().FirstAsync(q => q.OwnerLogin == "nunittestingadmin");
            subquestionTemplateNUnit.QuestionTemplateId = subquestionTemplateNUnit.QuestionTemplate.QuestionTemplateId;
            subquestionTemplateNUnit.OwnerLogin = "nunittestingadmin";
            await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit, null, null);

            TestResult testResult = new TestResult();
            testResult.TestTemplate = await GetTestTemplateDbSet().FirstAsync(t => t.Title == "postest" && t.OwnerLogin == "nunittestingadmin");
            testResult.TestTemplateId = testResult.TestTemplate.TestTemplateId;
            testResult.TimeStamp = DateTime.Now;
            testResult.Student = await dataFunctions.GetStudentByLogin("nunittestingstudent");
            testResult.StudentLogin = testResult.Student.Login;
            testResult.OwnerLogin = testResult.TestTemplate.OwnerLogin;
            testResult.IsTurnedIn = true;
            testResult.IsTestingData = true;
            List<QuestionResult> questionResults = new List<QuestionResult>();

            QuestionResult questionResult = new QuestionResult();
            questionResult.TestResult = testResult;
            questionResult.QuestionTemplate = await GetQuestionTemplateDbSet().FirstAsync(t => t.OwnerLogin == "nunittestingadmin");
            questionResult.OwnerLogin = questionResult.TestResult.OwnerLogin;
            questionResult.QuestionTemplateId = questionResult.QuestionTemplate.QuestionTemplateId;
            questionResult.TestResultId = questionResult.TestResult.TestResultId;
            List<SubquestionResult> subquestionResults = new List<SubquestionResult>();

            SubquestionResult subquestionResult = new SubquestionResult();
            subquestionResult.QuestionResult = questionResult;
            subquestionResult.SubquestionTemplate = await GetSubquestionTemplateDbSet().FirstAsync(t => t.OwnerLogin == "nunittestingadmin");
            subquestionResult.SubquestionTemplateId = subquestionResult.SubquestionTemplate.SubquestionTemplateId;
            subquestionResult.OwnerLogin = subquestionResult.QuestionResult.OwnerLogin;
            subquestionResult.StudentsAnswers = new string[] { "test1", "test3", "test2" };
            subquestionResult.StudentsPoints = 0;
            subquestionResult.DefaultStudentsPoints = 0;
            subquestionResult.AnswerCorrectness = 0.5;
            subquestionResult.AnswerStatus = EnumTypes.AnswerStatus.Incorrect;
            subquestionResults.Add(subquestionResult);

            questionResult.SubquestionResults = subquestionResults;
            questionResults.Add(questionResult);
            testResult.QuestionResults = questionResults;

            await dataFunctions.AddTestResult(testResult);

            return "NUnit data byla úspěšně přidána.";
        }

        /// <summary>
        /// Deletes data that is used during NUnit tests - users, students, templates, results, statistics..
        /// </summary>
        public async Task<string> DeleteNUnitData()
        {
            Subject? subject = await GetSubjectDbSet().FirstOrDefaultAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
            if(subject != null)
            {
                await dataFunctions.DeleteSubject(subject);
            }

            TestTemplate? testTemplate = await GetTestTemplateDbSet().FirstOrDefaultAsync(t => t.Title == "postest" && t.IsTestingData == true);
            if (testTemplate != null)
            {
                await dataFunctions.DeleteTestTemplate(testTemplate, null);
            }

            User? admin = await dataFunctions.GetUserDbSet().FirstOrDefaultAsync(s => s.Login == "nunittestingadmin");
            if(admin != null)
            {
                await dataFunctions.DeleteUser(admin);
            }

            User? teacher = await dataFunctions.GetUserDbSet().FirstOrDefaultAsync(s => s.Login == "nunittestingteacher");
            if (teacher != null)
            {
                await dataFunctions.DeleteUser(teacher);
            }

            Student? student = await dataFunctions.GetStudentDbSet().FirstOrDefaultAsync(s => s.Login == "nunittestingstudent");
            if(student != null)
            {
                await dataFunctions.DeleteStudent(student);
            }

            return "NUnit data byla úspěšně odebrána.";
        }
    }
}