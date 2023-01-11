using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using CsvHelper.Configuration.Attributes;

namespace DomainModel
{
    public class SubquestionTemplateRecord
    {
        //[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        //public int Id { get; set; }
        [Ignore]
        public string SubquestionIdentifier { get; set; }
        [Ignore]
        public string QuestionNumberIdentifier { get; set; }
        [Ignore]
        public SubquestionTemplate SubquestionTemplate { get; set; }
        [Ignore]
        public User Owner { get; set; } = default!;
        [Ignore]
        public string OwnerLogin { get; set; } = default!;
        public double SubquestionTypeAveragePoints { get; set; }
        public double CorrectAnswersShare { get; set; }
        public double SubjectAveragePoints { get; set; }
        public int ContainsImage { get; set; }
        public int NegativePoints { get; set; }
        public double? MinimumPointsShare { get; set; }
        public double? SubquestionPoints { get; set; }
    }
}
