using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class SubquestionTemplateRecord
    {
        //[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        //public int Id { get; set; }
        public string SubquestionIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }
        public SubquestionTemplate SubquestionTemplate { get; set; }
        public User Owner { get; set; } = default!;
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
