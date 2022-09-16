namespace DomainModel
{
    public class QuestionResult
    {
        public string TestResultIdentifier { get; set; } = default!;
        public string QuestionNumberIdentifier { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public TestResult TestResult { get; set; } = default!;
        public QuestionTemplate QuestionTemplate { get; set; } = default!;
        public ICollection<SubquestionResult> SubquestionResultList { get; set; } = default!;
    }
}