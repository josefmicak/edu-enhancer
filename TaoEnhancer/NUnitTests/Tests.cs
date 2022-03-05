using Common;
using Common.Class;
using DataLayer;
using NUnit.Framework;
using Moq;
using System.Xml;
using ViewLayer.Controllers;
using NUnitTestProject;

namespace NUnitTests
{
    [TestFixture]
    public class Tests
    {
        private static readonly object[] testNameIdentifier =
        {
            // Test paths that should be found
            new object[] { "mujtest", true, null },
            new object[] { "postest", true, null },
            new object[] { "viceparu", true, null },
            new object[] { "vsbtest", true, null },

            // Test paths that should not be found
            new object[] { "faketest", false, Exceptions.TestPathNotFoundException },
            new object[] { "nottest", false, Exceptions.TestPathNotFoundException },
            new object[] { "testoftest", false, Exceptions.TestPathNotFoundException },
            new object[] { "ufotest", false, Exceptions.TestPathNotFoundException }
        };

        // Checks whether a test that should be loaded is actually loaded
        [Test]
        [TestCaseSource(nameof(testNameIdentifier))]
        public void TestLoadNUnitTest(string testNameIdentifier, bool shouldFileBeFound, Exception expectedException)
        {
            TestDelegate testDelegate = () => new TestData().Load(testNameIdentifier);

            if (Directory.Exists(Settings.GetTestPath(testNameIdentifier)))
            {
                string testNumberIdentifier = new TestData().GetTestNumberIdentifier(testNameIdentifier);
                if (File.Exists(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier)))
                {
                    Assert.That(shouldFileBeFound, Is.EqualTo(true));
                    Assert.DoesNotThrow(testDelegate);
                }
            }
            else
            {
                Assert.That(shouldFileBeFound, Is.EqualTo(false));
                Assert.That(testDelegate, Throws.Exception.Message.EqualTo(expectedException.Message));
            }
        }

        private static readonly object[] testPointsDetermined =
        {
            // Only viceparu test should have points determined (by the current data)
            new object[] { "mujtest", 0, false, false },
            new object[] { "postest", 39, false, true },
            new object[] { "viceparu", 12, true, false },
            new object[] { "vsbtest", 51, false, false }
        };

        // Checks whether a test points match the expected value and whether the test that should have points determined actually has them determined
        [Test]
        [TestCaseSource(nameof(testPointsDetermined))]
        public void TestPointsNUnitTest(string testNameIdentifier, int expectedTestPoints, bool shouldTestPointsBeDetermined, bool expectedNegativePoints)
        {
            string testNumberIdentifier = new TestData().GetTestNumberIdentifier(testNameIdentifier);
            List<(string, string, string, string, int, bool)> itemParameters = new TestController().LoadItemInfo(testNameIdentifier, testNumberIdentifier);
            (int points, bool pointsDetermined) testPoints = new TestController().GetTestPoints(itemParameters);

            Assert.That(expectedTestPoints, Is.EqualTo(testPoints.points));

            if (testPoints.pointsDetermined)
            {
                Assert.That(shouldTestPointsBeDetermined, Is.EqualTo(true));
            }
            else
            {
                Assert.That(shouldTestPointsBeDetermined, Is.EqualTo(false));
            }

            bool negativePoints = new ItemController().NegativePoints(testNameIdentifier, testNumberIdentifier);

            if (negativePoints)
            {
                Assert.That(expectedNegativePoints, Is.EqualTo(true));
            }
            else
            {
                Assert.That(expectedNegativePoints, Is.EqualTo(false));
            }
        }

        private static readonly object[] TestData =
        {
            // Title set, Label set, Subitems: 1
            new object[] { "postestcopy", "", "i164458888823442", true, true, 1 },

            // Title set, Label not set, Subitems: 1
            new object[] { "postestcopy", "", "i1644589764485444", true, false, 1 },

            // Title not set, Label set, Subitems: 1
            new object[] { "postestcopy", "", "i16445865788370423", false, true, 1 },

            // Title not set, Label not set, Subitems: 1
            new object[] { "postestcopy", "", "i16445882115493440", false, false, 1 },

            // Title set, Label set, Subitems: 1
            new object[] { "postestcopy", "", "i16445883974289441", true, true, 2 },

            // Title set, Label not set, Subitems: 1
            new object[] { "postestcopy", "", "i16445890213918443", true, false, 2 },

            // Title not set, Label set, Subitems: 1
            new object[] { "postestcopy", "", "i16445900841567445", false, true, 1 }
        };

        [Test]//Má každý item přidělený label?
        [TestCaseSource((nameof(TestData)))]
        public void TitleLabelSubitemsSetNUnitTest(string testNameIdentifier, string itemNameIdentifier, string itemNumberIdentifier, bool titleSet, bool labelSet, int amountOfSubitems)
        {
            var mockItemController = new Mock<IFilePathManager>();
            mockItemController.Setup(p => p.GetFilePath(testNameIdentifier, itemNumberIdentifier)).Returns(GetFilePath(testNameIdentifier, itemNumberIdentifier));
            ItemController itemController = new ItemController(mockItemController.Object);

            (string, string, string, string, int) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);

            if (itemParameters.Item3.Length > 0)
            {
                Assert.That(titleSet, Is.EqualTo(true));
            }
            else
            {
                Assert.That(titleSet, Is.EqualTo(false));
            }

            if (itemParameters.Item4.Length > 0)
            {
                Assert.That(labelSet, Is.EqualTo(true));
            }
            else
            {
                Assert.That(labelSet, Is.EqualTo(false));
            }

            Assert.That(amountOfSubitems, Is.EqualTo(itemParameters.Item5));
        }

        public string GetFilePath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier);
        }
    }
}