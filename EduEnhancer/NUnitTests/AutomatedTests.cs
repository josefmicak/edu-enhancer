using ArtificialIntelligenceTools;
using BusinessLayer;
using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NUnitTests
{
    public class AutomatedTests
    {
        private IWebDriver driver;
        private IJavaScriptExecutor js;
        private readonly IConfiguration _configuration;

        [SetUp]
        public void SetUp()
        {
            driver = new ChromeDriver("../../../../");
            js = (IJavaScriptExecutor)driver;
        }

        [TearDown]
        protected void TearDown()
        {
            driver.Quit();
        }

        public async Task<CourseContext> GetCourseContext()
        {

            var options = new DbContextOptionsBuilder<CourseContext>()
            .UseInMemoryDatabase(databaseName: "EduEnhancerDB")
            .Options;

            using (var _context = new CourseContext(options))
            {
                _context.Database.EnsureDeleted();

                //first test

                _context.Users.Add(new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                });
                _context.Students.Add(new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                });
                _context.SaveChanges();

                _context.Subjects.Add(new Subject
                {
                    Abbreviation = "POS",
                    Name = "Počítačové sítě",
                    Guarantor = _context.Users.First(u => u.Login == "nunittestingadmin"),
                    GuarantorLogin = "nunittestingadmin",
                    IsTestingData = true,
                    Students = new List<Student> { _context.Students.First() }
                });
                _context.SaveChanges();

                DataFunctions dataFunctions = new DataFunctions(_context);
                TestTemplate testTemplateNUnit = new TestTemplate();
                testTemplateNUnit.Title = "postest";
                testTemplateNUnit.MinimumPoints = 0;
                testTemplateNUnit.StartDate = new DateTime(2022, 12, 30, 00, 00, 00);
                testTemplateNUnit.EndDate = new DateTime(2022, 12, 31, 00, 00, 00);
                testTemplateNUnit.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit.Owner = await _context.Users.FirstAsync(u => u.Login == "nunittestingadmin");
                testTemplateNUnit.OwnerLogin = "nunittestingadmin";
                testTemplateNUnit.IsTestingData = true;
                testTemplateNUnit.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit);

                QuestionTemplate questionTemplateNUnit = new QuestionTemplate();
                questionTemplateNUnit.Title = "NUnit title";
                questionTemplateNUnit.OwnerLogin = "nunittestingadmin";
                questionTemplateNUnit.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.Title == "postest" && t.OwnerLogin == "nunittestingadmin");
                questionTemplateNUnit.SubquestionTemplates = new List<SubquestionTemplate>();
                await dataFunctions.AddQuestionTemplate(questionTemplateNUnit);

                SubquestionTemplate subquestionTemplateNUnit = new SubquestionTemplate();
                subquestionTemplateNUnit.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
                subquestionTemplateNUnit.SubquestionText = "";
                subquestionTemplateNUnit.PossibleAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit.CorrectAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit.SubquestionPoints = 10;
                subquestionTemplateNUnit.CorrectChoicePoints = 10 / 3;
                subquestionTemplateNUnit.DefaultWrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit.WrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.OwnerLogin == "nunittestingadmin");
                subquestionTemplateNUnit.QuestionTemplateId = subquestionTemplateNUnit.QuestionTemplate.QuestionTemplateId;
                subquestionTemplateNUnit.OwnerLogin = "nunittestingadmin";
                await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit, null, null);

                TestResult testResult = new TestResult();
                testResult.TestTemplate = await _context.TestTemplates.FirstAsync();
                testResult.TestTemplateId = testResult.TestTemplate.TestTemplateId;
                testResult.TimeStamp = DateTime.Now;
                testResult.Student = await _context.Students.FirstAsync();
                testResult.StudentLogin = testResult.Student.Login;
                testResult.OwnerLogin = testResult.TestTemplate.OwnerLogin;
                testResult.IsTurnedIn = true;
                testResult.IsTestingData = true;
                testResult.QuestionResults = new List<QuestionResult>();
                _context.TestResults.Add(testResult);
                await _context.SaveChangesAsync();

                QuestionResult questionResult = new QuestionResult();
                questionResult.TestResult = await _context.TestResults.FirstAsync();
                questionResult.QuestionTemplate = await _context.QuestionTemplates.FirstAsync();
                questionResult.OwnerLogin = questionResult.TestResult.OwnerLogin;
                questionResult.QuestionTemplateId = questionResult.QuestionTemplate.QuestionTemplateId;
                questionResult.TestResultId = questionResult.TestResult.TestResultId;
                questionResult.SubquestionResults = new List<SubquestionResult>();
                _context.QuestionResults.Add(questionResult);
                await _context.SaveChangesAsync();

                SubquestionResult subquestionResult = new SubquestionResult();
                subquestionResult.QuestionResult = await _context.QuestionResults.Include(q => q.QuestionTemplate).FirstAsync();
                subquestionResult.QuestionResultId = subquestionResult.QuestionResult.QuestionResultId;
                subquestionResult.QuestionTemplateId = subquestionResult.QuestionResult.QuestionTemplateId;
                subquestionResult.TestResultId = await _context.TestResults.Select(t => t.TestResultId).FirstAsync();
                subquestionResult.SubquestionTemplate = await _context.SubquestionTemplates.FirstAsync();
                subquestionResult.SubquestionTemplateId = subquestionResult.SubquestionTemplate.SubquestionTemplateId;
                subquestionResult.OwnerLogin = subquestionResult.QuestionResult.OwnerLogin;
                subquestionResult.StudentsAnswers = new string[] { "test1", "test3", "test2" };
                subquestionResult.StudentsPoints = 0;
                subquestionResult.DefaultStudentsPoints = 0;
                subquestionResult.AnswerCorrectness = 0.5;
                subquestionResult.AnswerStatus = EnumTypes.AnswerStatus.Incorrect;
                _context.SubquestionResults.Add(subquestionResult);
                await _context.SaveChangesAsync();
            }

            return new CourseContext(options);
        }

        private static readonly object[] subquestionResultPoints =
{
            new object[] { "10.01", false, "Chyba: příliš vysoký počet bodů. Nejvyšší počet bodů, které může za tuto podotázku student obdržet, je 10."},
            new object[] { "10", true, "Studentův počet bodů byl úspěšně změněn."},
            new object[] { "-10", true, "Studentův počet bodů byl úspěšně změněn."},
            new object[] { "-10.01", false, "Chyba: příliš nízký počet bodů. Nejnížší počet bodů, které může za tuto podotázku student obdržet, je -10." }
        };

        [Test]
        [TestCaseSource(nameof(subquestionResultPoints))]
        public void SubquestionResultPointsTest(string studentsPoints, bool pointsChangeExpected, string expectedOutcome)
        {
            double defaultStudentsPoints = 0;
            double defaultSubquestionTemplatePoints = 10;
            driver.Navigate().GoToUrl("https://localhost:7026");
            var selectedUserLogin = driver.FindElement(By.Id("selectedUserLogin"));
            selectedUserLogin.Click();
            var selectElement = new SelectElement(selectedUserLogin);
            selectElement.SelectByText("Email: EmailAdmin, login: nunittestingadmin");

            driver.FindElement(By.Id("testing-sign-in")).Click();
            driver.FindElement(By.Id("manage-solved-tests")).Click();
            driver.FindElement(By.Id("manage-solved-test-0")).Click();
            driver.FindElement(By.Id("manage-solved-question-0")).Click();
            var oldSubquestionPoints = driver.FindElement(By.Id("students-points")).GetAttribute("value");
            Assert.That(oldSubquestionPoints.ToString(), Is.EqualTo(defaultStudentsPoints.ToString()));

            IWebElement oldStudentsPointsInput = driver.FindElement(By.Id("students-points"));
            oldStudentsPointsInput.Clear();
            oldStudentsPointsInput.SendKeys(studentsPoints);
            driver.FindElement(By.Id("save-students-points")).Click();

            var subquestionResultPointsMessage = driver.FindElement(By.Id("subquestion-result-points-message")).Text;
            Assert.That(subquestionResultPointsMessage, Is.EqualTo(expectedOutcome));

            var newSubquestionPoints = driver.FindElement(By.Id("students-points")).GetAttribute("value");
            var studentsQuestionPoints = driver.FindElement(By.Id("students-question-points")).Text;
            var studentsSubquestionPoints = driver.FindElement(By.Id("students-subquestion-points")).Text;
            if (pointsChangeExpected)
            {
                Assert.That(newSubquestionPoints.ToString(), Is.EqualTo(studentsPoints));
                Assert.That(studentsQuestionPoints, Is.EqualTo(studentsPoints + " / " + defaultSubquestionTemplatePoints.ToString()));
                Assert.That(studentsSubquestionPoints, Is.EqualTo(studentsPoints + " / " + defaultSubquestionTemplatePoints.ToString()));
            }
            else
            {
                Assert.That(newSubquestionPoints.ToString(), Is.EqualTo(defaultStudentsPoints.ToString()));
                Assert.That(studentsQuestionPoints, Is.EqualTo(defaultStudentsPoints.ToString() + " / " + defaultSubquestionTemplatePoints.ToString()));
                Assert.That(studentsSubquestionPoints, Is.EqualTo(defaultStudentsPoints.ToString() + " / " + defaultSubquestionTemplatePoints.ToString()));
            }

            //restore old value
            IWebElement newStudentsPointsInput = driver.FindElement(By.Id("students-points"));
            newStudentsPointsInput.Clear();
            newStudentsPointsInput.SendKeys(defaultStudentsPoints.ToString());
            driver.FindElement(By.Id("save-students-points")).Click();
            Assert.That(oldSubquestionPoints.ToString(), Is.EqualTo(defaultStudentsPoints.ToString()));
        }
    }
}
