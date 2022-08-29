namespace DomainModel
{
    public class QuestionResult
    {
        public string TestResultIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }
        public string OwnerLogin { get; set; }
        public TestResult TestResult { get; set; }
        public QuestionTemplate QuestionTemplate { get; set; }
        public ICollection<SubquestionResult> SubquestionResultList { get; set; }
    }
}