namespace DomainModel
{
    public class TestResult
    {
        public string TestResultIdentifier { get; set; }
        public string TestNameIdentifier { get; set; }
        public string StudentIdentifier { get; set; }
        public string TimeStamp { get; set; }
        public List<QuestionResult> QuestionResultList { get; set; }
    }
}