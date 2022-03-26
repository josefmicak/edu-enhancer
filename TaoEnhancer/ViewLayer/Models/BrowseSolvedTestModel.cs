namespace ViewLayer.Models
{
    public class BrowseSolvedTestModel : PageModel
    {
        private string studentIdentifier;
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private string deliveryExecutionIdentifier;
        private (string, string, string title, int amountOfItems) testParameters;
        private List<(string, string, string, string)> questionList = new List<(string, string, string, string)>();
        private List<(string, string, string, string, int, bool)> itemParameters = new List<(string, string, string, string, int, bool)>();
        private (int testPoints, bool testPointsDetermined) testPoints;
        private (List<(double questionResultPoints, bool questionResultPointsDetermined)> studentsPoints, int errorMessageNumber) questionResultPoints;
        private double totalStudentsPoints;
        private (string resultIdentifier, string resultTimestamp, string studentName, string studentLogin, string) resultParameters;

        public string StudentIdentifier { get { return studentIdentifier; } set { studentIdentifier = value; } }
        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public string DeliveryExecutionIdentifier { get { return deliveryExecutionIdentifier; } set { deliveryExecutionIdentifier = value; } }
        public (string, string, string title, int amountOfItems) TestParameters { get { return testParameters; } set { testParameters = value; } }
        public List<(string, string, string, string)> QuestionList { get { return questionList; } set { questionList = value; } }
        public List<(string, string, string, string, int, bool)> ItemParameters { get { return itemParameters; } set { itemParameters = value; } }
        public (int testPoints, bool testPointsDetermined) TestPoints { get { return testPoints; } set { testPoints = value; } }
        public (List<(double questionResultPoints, bool questionResultPointsDetermined)> studentsPoints, int errorMessageNumber) QuestionResultPoints { get { return questionResultPoints; } set { questionResultPoints = value; } }
        public double TotalStudentsPoints { get { return totalStudentsPoints; } set { totalStudentsPoints = value; } }
        public (string resultIdentifier, string resultTimestamp, string studentName, string studentLogin, string) ResultParameters { get { return resultParameters; } set { resultParameters = value; } }
    }
}