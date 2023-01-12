using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class QuestionTemplate
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int QuestionTemplateId { get; set; }
        public string Title { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public TestTemplate TestTemplate { get; set; } = default!;
        public ICollection<SubquestionTemplate> SubquestionTemplateList { get; set; } = default!;
    }
}