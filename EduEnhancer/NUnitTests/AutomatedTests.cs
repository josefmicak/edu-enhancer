using ArtificialIntelligenceTools;
using BusinessLayer;
using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using NUnit.Framework.Internal;
using NUnit.Framework;
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

        [SetUp]
        public void SetUp()
        {
            driver = new ChromeDriver("../../../../");//the driver is located inside the "bin" folder in the NUnitTests project folder
            js = (IJavaScriptExecutor)driver;
        }

        [TearDown]
        protected void TearDown()
        {
            driver.Quit();
        }

        private static readonly object[] _subquestionResultPoints =
{
            new object[] { "10.01", false, "Chyba: příliš vysoký počet bodů. Nejvyšší počet bodů, které může za tuto podotázku student obdržet, je 10."},
            new object[] { "10", true, "Studentův počet bodů byl úspěšně změněn."},
            new object[] { "-10", true, "Studentův počet bodů byl úspěšně změněn."},
            new object[] { "-10.01", false, "Chyba: příliš nízký počet bodů. Nejnížší počet bodů, které může za tuto podotázku student obdržet, je -10." }
        };

        [Test]
        [TestCaseSource(nameof(_subquestionResultPoints))]
        public void SubquestionResultPointsTest(string studentsPoints, bool pointsChangeExpected, string expectedOutcome)
        {
            double defaultStudentsPoints = 0;
            double defaultSubquestionTemplatePoints = 10;
            driver.Navigate().GoToUrl(Config.GetURL());
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

        private static readonly object[] _userParameters =
        {
            //unique student - successfully added
            new object[] { "Name", "Surname", "UniqueStudentLogin", "UniqueStudent@Email.com", "student", true, 
                "Student s loginem \"UniqueStudentLogin\" úspěšně přidán." },
            //unique teacher - successfully added
            new object[] { "Name", "Surname", "UniqueTeacherLogin", "UniqueTeacher@Email.com", "teacher", true,
                "Učitel byl úspěšně přidán." },
            //student with no first name - error message is sent back
            new object[] { "", "Surname", "UniqueStudentLogin", "UniqueStudent@Email.com", "student", false,
                "Chyba: je nutné vyplnit všechna pole." },
            //student with invalid email - error message is sent back
            new object[] { "", "Surname", "UniqueStudentLogin", "InvalidStudentEmail", "student", false,
                 "Chyba: \"InvalidStudentEmail\" není správně formátovaná emailová adresa." },
            //student with login of an existing user - error message is sent back
            new object[] { "Name", "Surname", "nunittestingstudent", "UniqueStudent@Email.com", "student", false,
                "Chyba: uživatel s loginem \"nunittestingstudent\" již existuje." },
            //student with email of an existing user - error message is sent back
            new object[] { "Name", "Surname", "UniqueStudentLogin", "EmailAdmin", "student", false,
                "Chyba: uživatel s emailem \"LoginAdmin\" již existuje." },
            //teacher with login of an existing user - error message is sent back
            new object[] { "Name", "Surname", "nunittestingteacher", "UniqueTeacher@Email.com", "teacher", false,
                "Chyba: uživatel s loginem \"nunittestingteacher\" již existuje." },
             //teacher with email of an existing user - error message is sent back
            new object[] { "Name", "Surname", "nunittestingteacher", "EmailStudent", "teacher", false,
                "Chyba: uživatel s emailem \"EmailStudent\" již existuje." }
        };

        [Test]
        [TestCaseSource(nameof(_userParameters))]
        public void RegistrationApprovalTest(string firstName, string lastName, string login, string email, string role, 
            bool newUserExpected, string expectedOutcome)
        {
            driver.Navigate().GoToUrl(Config.GetURL());
            var selectedUserLogin = driver.FindElement(By.Id("selectedUserLogin"));
            selectedUserLogin.Click();
            var selectElement = new SelectElement(selectedUserLogin);
            selectElement.SelectByText("Email: EmailAdmin, login: nunittestingadmin");

            driver.FindElement(By.Id("testing-sign-in")).Click();
            driver.FindElement(By.Id("manage-user-list")).Click();

            IWebElement firstNameInput = driver.FindElement(By.Id(role + "FirstName"));
            firstNameInput.Clear();
            firstNameInput.SendKeys(firstName);

            IWebElement lastNameInput = driver.FindElement(By.Id(role + "LastName"));
            lastNameInput.Clear();
            lastNameInput.SendKeys(lastName);

            IWebElement loginInput = driver.FindElement(By.Id(role + "Login"));
            loginInput.Clear();
            loginInput.SendKeys(login);

            IWebElement emailInput = driver.FindElement(By.Id(role + "Email"));
            emailInput.Clear();
            emailInput.SendKeys(email);

            driver.FindElement(By.Id("save-" + role)).Click();

            if (newUserExpected)
            {
                var userMessage = driver.FindElement(By.Id(role + "-message")).Text;
                Assert.That(userMessage, Is.EqualTo(expectedOutcome));
            }

            //restore old state - delete added user, confirm whether he's been deleted
            if (newUserExpected)
            {
                driver.FindElement(By.Id("delete-" + login)).Click();
                driver.FindElement(By.Id("confirm-action-yes")).Click();
                var userMessage = driver.FindElement(By.Id(role + "-message")).Text;
                if (role == "student")
                {
                    Assert.That(userMessage, Is.EqualTo("Student úspěšně smazán."));
                }
                else if (role == "teacher")
                {
                    Assert.That(userMessage, Is.EqualTo("Učitel úspěšně smazán."));
                }
            }
        }

        private static readonly object[] _subquestionTemplatePoints =
        {
            //non-positive value entered for subquestion points - error message is sent back
            new object[] { "0", "-0,01", false, "Chyba: nekompletní zadání podotázky (body)." },
            //valid values entered for both subquestion and wrong choice points - points are successfully changed
            new object[] { "0,01", "-0,01", true, "Zadání podotázky bylo úspěšně upraveno." },
            //wrong choice points equals less than subquestion points * (-1) - error message is sent back
            new object[] { "0,01", "-0,02", false, "Chyba: nejmenší možný počet bodů za špatnou volbu je -0,01." },
            //non-negative entered for wrong choice points - error message is sent back
            new object[] { "0,01", "0", false, "Chyba: za špatnou volbu nemůže být udělen nezáporný počet bodů." }
        };


        [Test]
        [TestCaseSource(nameof(_subquestionTemplatePoints))]
        public void SubquestionTemplatePointsTest(string subquestionPoints, string wrongChoicePoints, bool pointsChangeExpected, string expectedOutcome)
        {
            double defaultQuestionTemplatePoints = 10;
            double defaultSubquestionTemplatePoints = 10;
            double defaultWrongChoicePoints = -10;
            driver.Navigate().GoToUrl(Config.GetURL());
            var selectedUserLogin = driver.FindElement(By.Id("selectedUserLogin"));
            selectedUserLogin.Click();
            var selectElement = new SelectElement(selectedUserLogin);
            selectElement.SelectByText("Email: EmailAdmin, login: nunittestingadmin");
            
            driver.FindElement(By.Id("testing-sign-in")).Click();
            driver.FindElement(By.Id("test-template-list-link")).Click();
            driver.FindElement(By.Id("open-test-template-0")).Click();
            driver.FindElement(By.Id("open-question-template-0")).Click();
            driver.FindElement(By.Id("edit-subquestion-template")).Click();
            
            IWebElement oldSubquestionPointsInput = driver.FindElement(By.Id("subquestion-points"));
            oldSubquestionPointsInput.Clear();
            oldSubquestionPointsInput.SendKeys(subquestionPoints);

            driver.FindElement(By.Id("wrongChoicePoints_manual_radio")).Click();
            IWebElement oldWrongChoicePointsManualInput = driver.FindElement(By.Id("wrongChoicePoints_manual"));
            oldWrongChoicePointsManualInput.Clear();
            oldWrongChoicePointsManualInput.SendKeys(wrongChoicePoints);

            driver.FindElement(By.Id("subquestion-add")).Click();
            //Thread.Sleep(2000);
            if (pointsChangeExpected)
            {
                var questionPointsString = driver.FindElement(By.Id("question-points-string")).Text;
                Assert.That(questionPointsString, Is.EqualTo(subquestionPoints));
                var subquestionPointsString = driver.FindElement(By.Id("subquestion-points-string")).Text;
                Assert.That(subquestionPointsString, Is.EqualTo(subquestionPoints));
                var questionTemplateMessage = driver.FindElement(By.Id("question-template-message")).Text;
                Assert.That(questionTemplateMessage, Is.EqualTo(expectedOutcome));
            }
            else
            {
                var newSubquestionPoints = driver.FindElement(By.Id("subquestion-points")).GetAttribute("value");
                Assert.That(newSubquestionPoints, Is.EqualTo(defaultSubquestionTemplatePoints.ToString()));
                var newWrongChoicePoints = driver.FindElement(By.Id("wrongChoicePoints_manual")).GetAttribute("value");
                Assert.That(newWrongChoicePoints, Is.Not.EqualTo(wrongChoicePoints.ToString()));
                var editSubquestionTemplateMessage = driver.FindElement(By.Id("edit-subquestion-template-message")).Text;
                Assert.That(editSubquestionTemplateMessage, Is.EqualTo(expectedOutcome));
            }

            //restore old value
            if (pointsChangeExpected)
            {
                driver.FindElement(By.Id("edit-subquestion-template")).Click();
                IWebElement newSubquestionPointsInput = driver.FindElement(By.Id("subquestion-points"));
                newSubquestionPointsInput.Clear();
                newSubquestionPointsInput.SendKeys(defaultSubquestionTemplatePoints.ToString());

                driver.FindElement(By.Id("wrongChoicePoints_manual_radio")).Click();
                IWebElement newWrongChoicePointsManualInput = driver.FindElement(By.Id("wrongChoicePoints_manual"));
                newWrongChoicePointsManualInput.Clear();
                newWrongChoicePointsManualInput.SendKeys(defaultWrongChoicePoints.ToString());

                driver.FindElement(By.Id("subquestion-add")).Click();
                var questionPointsStringNew = driver.FindElement(By.Id("question-points-string")).Text;
                Assert.That(questionPointsStringNew, Is.EqualTo(defaultQuestionTemplatePoints.ToString()));
                var subquestionPointsStringNew = driver.FindElement(By.Id("subquestion-points-string")).Text;
                Assert.That(subquestionPointsStringNew, Is.EqualTo(defaultSubquestionTemplatePoints.ToString()));
                var questionTemplateMessageNew = driver.FindElement(By.Id("question-template-message")).Text;
                Assert.That(questionTemplateMessageNew, Is.EqualTo(expectedOutcome));
            }
        }
    }
}
