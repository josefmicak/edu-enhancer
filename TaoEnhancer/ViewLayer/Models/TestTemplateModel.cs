namespace ViewLayer.Models
{
    public class TestTemplateModel : PageModel
    {
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private (string, string, string title, int amountOfItems) testParameters;
        private List<(string, string, string, string)> questionList = new List<(string, string, string, string)>();
        private List<(string, string, string, string, int, bool)> itemParameters = new List<(string, string, string, string, int, bool)>();
        private (int testPoints, bool testPointsDetermined) testPoints;
        private bool negativePoints;
        private string negativePointsOption;

        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public (string, string, string title, int amountOfItems) TestParameters { get { return testParameters; } set { testParameters = value; } }
        public List<(string, string, string, string)> QuestionList { get { return questionList; } set { questionList = value; } }
        public List<(string, string, string, string, int, bool)> ItemParameters { get { return itemParameters; } set { itemParameters = value; } }
        public (int testPoints, bool testPointsDetermined) TestPoints { get { return testPoints; } set { testPoints = value; } }
        public bool NegativePoints { get { return negativePoints; } set { negativePoints = value; } }
        public string NegativePointsOption { get { return negativePointsOption; } set { negativePointsOption = value; } }
    }
}