using CsvHelper.Configuration.Attributes;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionResultRecord
    {
        [Ignore]
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int SubquestionTemplateRecordId { get; set; }
        [Ignore]
        public int SubquestionTemplateId { get; set; }
        [Ignore]
        public int QuestionTemplateId { get; set; }
        [Ignore]
        public int TestResultId { get; set; }
        [Ignore]
        public SubquestionResult SubquestionResult { get; set; }
        [Ignore]
        public User Owner { get; set; } = default!;
        [Ignore]
        public int Id { get; set; }
        public string OwnerLogin { get; set; } = default!;
        public double SubquestionTypeAveragePoints { get; set; }
        public double AnswerCorrectness { get; set; }
        public double SubjectAveragePoints { get; set; }
        public int ContainsImage { get; set; }
        public int NegativePoints { get; set; }
        public double? MinimumPointsShare { get; set; }
        public double StudentsPoints { get; set; }
    }
}
