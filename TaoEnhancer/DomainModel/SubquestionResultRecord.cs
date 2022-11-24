using CsvHelper.Configuration.Attributes;

namespace DomainModel
{
    public class SubquestionResultRecord
    {
        [Ignore]
        public string SubquestionIdentifier { get; set; }
        [Ignore]
        public string QuestionNumberIdentifier { get; set; }
        [Ignore]
        public string TestResultIdentifier { get; set; }
        [Ignore]
        public SubquestionResult SubquestionResult { get; set; }
        [Ignore]
        public User Owner { get; set; } = default!;
        [Ignore]
        public string OwnerLogin { get; set; } = default!;
        public double SubquestionTypeAveragePoints { get; set; }
        public double AnswerCorrectness { get; set; }
        public double SubjectAveragePoints { get; set; }
        public int ContainsImage { get; set; }
        public int NegativePoints { get; set; }
        public double? MinimumPointsShare { get; set; }
        public double? StudentsPoints { get; set; }
    }
}
